from word2vec_extended import Word2VecExtended
from gensim.models.word2vec import Vocab, logger
from os.path import isfile as file_exists
import re, gzip, pickle, threading, time
try:
	from queue import Queue
except ImportError:
	from Queue import Queue
from six import iteritems, itervalues
from six.moves import xrange
from numpy import zeros, float32 as REAL, random, get_include, array, empty, sqrt, newaxis, argsort, dot, ndarray, hstack
from gensim import utils, matutils

from six import string_types

import pyximport
pyximport.install(setup_args={"include_dirs": get_include()})
from .pvdm_inner import train_sentence_pvdm, train_sentence_batch_pvdm, train_sentence_batch_pvdm_self_predict, train_sentence_batch_pvdm_skipgram

PaddingWord = "**PADDING_WORD**"
MAX_BATCHSIZE = 1000

class Paragraph:
	def __init__(self):
		self.index = None

class PVDM(Word2VecExtended):

	def __init__(self, vocabulary = None, random_window = False, scale_updates = False, self_predict = 0, batchsize = 100, symmetric_window = True, oov_word = True, min_count = 5, paragraph_size = 400, concatenate = True, sentences=None, size=400, alpha=0.035, window=5, seed=1, workers=1, min_alpha=0.0001, decay = True, vocab_report_frequency = 10000):
		"""

		PVDM model for training and learning context paragraphs for sentiment and topic
		analysis, or information retrieval.

		This method uses hierarchical softmax, word2vec, and word windows to obtain an
		unsupervised model for these paragraphs and their context [1]

		[1] Quoc Le and Tomas Mikolov, "Distributed Representations of Sentences and Documents," ICML 2014.

		TODO:

		- add synparagraph for updating paragraph positions
		- store paragraph size
		- build record of paragraph index (without code)
		- update training function accordingly.

		"""

		if batchsize > MAX_BATCHSIZE:
			raise AssertionError("Maximum batch size is %d." % (MAX_BATCHSIZE))

		self.batchsize = int(batchsize) if batchsize > 0 else 1
		self.symmetric_window = symmetric_window
		self.scale_updates = scale_updates

		self.vocab = {}  # mapping from a word (string) to a Vocab object
		self.paragraph_vocab = {}
		self.index2word = []  # map from a word's matrix index (int) to word (string)
		self.index2paragraph = [] # map from a paragraph's matrix index (int) to paragraph (string)

		self.layer1_size = int(size)
		self.paragraph_size = int(paragraph_size)

		self.concatenate = concatenate
		self.random_window = random_window

		if size % 4 != 0:
			logger.warning("consider setting layer size to a multiple of 4 for greater performance")

		self.alpha = float(alpha)
		self.window = int(window)
		self.weight_decay = decay
		self.seed = seed
		self.hs = True
		self.negative = False

		self.self_predict = self_predict

		self.min_count = min_count
		self.workers   = workers
		self.min_alpha = min_alpha

		if self.concatenate:
			# the logistic regression layer for hierarchical softmax deals
			# first with the paragraph dimensions, then with window * 2
			# words:
			if self.symmetric_window:
				self.logistic_regression_size = self.paragraph_size + self.window * 2 * self.layer1_size
			else:
				self.logistic_regression_size = self.paragraph_size + self.window * 1 * self.layer1_size
		else:
			# the logistic regression layer for hierarchical softmax deals first
			# with the paragraph dimensions, then with the average of the
			# 2 * window words:
			self.logistic_regression_size = self.layer1_size + self.paragraph_size

		if self_predict > 0:
			self.training_function = train_sentence_batch_pvdm_self_predict if self_predict == 1 else train_sentence_batch_pvdm_skipgram
			self.logistic_regression_size = self.layer1_size
			self.true_paragraph_size = self.paragraph_size
			self.paragraph_size = 0
		else:
			self.training_function = train_sentence_batch_pvdm

		if sentences is not None:
			self.build_vocab(sentences, oov_word = oov_word, report_frequency = vocab_report_frequency)
			self.train(sentences) # maybe ?

	def reintroduce_multi_word_windows(self):
		if self.self_predict > 0:

			if self.concatenate:
				# the logistic regression layer for hierarchical softmax deals
				# first with the paragraph dimensions, then with window * 2
				# words:
				if self.symmetric_window:
					self.logistic_regression_size = self.paragraph_size + self.window * 2 * self.layer1_size
				else:
					self.logistic_regression_size = self.paragraph_size + self.window * 1 * self.layer1_size
			else:
				# the logistic regression layer for hierarchical softmax deals first
				# with the paragraph dimensions, then with the average of the
				# 2 * window words:
				self.logistic_regression_size = self.layer1_size + self.paragraph_size

			self.paragraph_size = self.true_paragraph_size

			self.training_function = train_sentence_batch_pvdm
			if self.symmetric_window:
				repeat_times = 2 * self.window
			else:
				repeat_times = 1 * self.window
			repeated_syn1 = self.syn1.repeat(repeat_times, axis=1) / repeat_times
			new_paragraph_syn1 = zeros([len(self.vocab), self.paragraph_size], dtype = REAL)
			stacked_syn1 =  hstack([new_paragraph_syn1, repeated_syn1]).astype(dtype = REAL)
			self.syn1 = stacked_syn1

			# reintroduce paragraph vectors:
			self.synparagraph = empty((len(self.paragraph_vocab), self.paragraph_size), dtype=REAL)
			# randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
			for i in xrange(len(self.paragraph_vocab)):
				self.synparagraph[i] = (random.rand(self.paragraph_size) - 0.5) / self.paragraph_size

			self.self_predict = 0
		else:
			raise AssertionError("Not self-predicting, cannot switch back to multi word windows. Predictor is already in multi word prediction mode.")

	def reset_weights(self):
		super().reset_weights()
		self.synparagraph = empty((len(self.paragraph_vocab), self.paragraph_size), dtype=REAL)
		# randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
		for i in xrange(len(self.paragraph_vocab)):
			self.synparagraph[i] = (random.rand(self.paragraph_size) - 0.5) / self.paragraph_size

	def init_sims(self, replace=False):
		"""
		Precompute L2-normalized vectors.

		If `replace` is set, forget the original vectors and only keep the normalized
		ones = saves lots of memory!

		Note that you **cannot continue training** after doing a replace. The model becomes
		effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

		"""
		super().init_sims(replace = replace)

		if getattr(self, 'synparagraphnorm', None) is None or replace:
			logger.info("precomputing L2-norms of word weight vectors")
			if replace:
				for i in range(self.synparagraph.shape[0]):
					self.synparagraph[i, :] /= sqrt((self.synparagraph[i, :] ** 2).sum(-1))
				self.synparagraphnorm = self.synparagraph
			else:
				self.synparagraphnorm = (self.synparagraph / sqrt((self.synparagraph ** 2).sum(-1))[..., newaxis]).astype(REAL)

	def most_similar_paragraph(self, positive=[], negative=[], topn=10):
		"""
		Find the top-N most similar paragraphs.

		"""
		self.init_sims()

		if isinstance(positive, string_types) and not negative:
			# allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
			positive = [positive]

		# add weights for each paragraph, if not already present; default to 1.0 for positive and -1.0 for negative paragraphs
		positive = [(paragraph, 1.0) if isinstance(paragraph, string_types + (ndarray,))
								else paragraph for paragraph in positive]
		negative = [(paragraph, -1.0) if isinstance(paragraph, string_types + (ndarray,))
								 else paragraph for paragraph in negative]

		# compute the weighted average of all words
		all_paragraphs, mean = set(), []
		for paragraph, weight in positive + negative:
			if isinstance(paragraph, ndarray):
				mean.append(weight * paragraph)
			elif paragraph in self.paragraph_vocab:
				mean.append(weight * self.synparagraphnorm[self.paragraph_vocab[paragraph].index])
				all_paragraphs.add(self.paragraph_vocab[paragraph].index)
			else:
				raise KeyError("paragraph '%s' not in vocabulary" % paragraph)
		if not mean:
			raise ValueError("cannot compute similarity with no input")
		mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

		dists = dot(self.synparagraphnorm, mean)
		if not topn:
			return dists
		best = argsort(dists)[::-1][:topn + len(all_paragraphs)]
		# ignore (don't return) words from the input
		result = [(self.index2paragraph[sim], float(dists[sim]), sim) for sim in best if sim not in all_paragraphs]
		return result[:topn]

	def train(self, sentences, total_words=None, word_count=0, paragraphs_only = False, vocab = None, paragraphs = None):
		"""
		Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
		Each sentence must be a list of utf8 strings.

		"""
		if paragraphs is None:
			paragraphs = self.synparagraph
		if vocab is None:
			vocab = self.paragraph_vocab

		if not self.vocab:
			raise RuntimeError("you must first build vocabulary before training the model")

		start, next_report = time.time(), [1.0]
		word_count, total_words = [word_count], total_words or sum(v.count for v in itervalues(self.vocab))
		jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
		lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)
		total_error = [0.0]

		def worker_train():
			"""Train the model, lifting lists of sentences from the jobs queue."""
			paragraph_work = zeros(self.paragraph_size, dtype=REAL)  # each thread must have its own work memory
			error = zeros(1, dtype = REAL)
			if self.concatenate:
				# word work here is for each individual word, so it has length logistic regression - para size
				word_work = zeros(self.logistic_regression_size - self.paragraph_size, dtype = REAL)
				neu1 = matutils.zeros_aligned(self.logistic_regression_size, dtype=REAL)
			else:
				# here word work is aggregated:
				word_work = zeros(self.layer1_size, dtype = REAL)
				neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)

			zeros(self.logistic_regression_size, dtype = REAL)
			while True:
				job = jobs.get()
				if job is None:  # data finished, exit
					break
				# update the learning rate before every job
				alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words)) if self.weight_decay else self.alpha
				# how many words did we train on? out-of-vocabulary (unknown) words do not count
				job_words = self.training_function(self, job, paragraphs, paragraphs_only, alpha, paragraph_work, word_work, neu1, error, len(job))

				with lock:
					# here we can store the scores for later plotting and viewing...
					word_count[0] += job_words

					elapsed = time.time() - start
					total_error[0] += error[0]
					if elapsed >= next_report[0]:
						logger.debug("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s," %
							(100.0 * word_count[0] / total_words, alpha, word_count[0] / elapsed if elapsed else 0.0))
						next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

		workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
		for thread in workers:
			thread.daemon = True  # make interrupting the process with ctrl+c easier
			thread.start()

		# convert input strings to Vocab objects, and paragraph to paragraph (Vocab) object:
		no_oov = (self.create_job(sentence,vocab) for sentence in sentences)
		for job_no, job in enumerate(utils.grouper(no_oov, self.batchsize)):
			logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
			jobs.put(job)
		logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
		for _ in xrange(self.workers):
			jobs.put(None)  # give the workers heads up that they can finish -- no more work!

		for thread in workers:
			thread.join()

		elapsed = time.time() - start
		logger.info("training on %i sentences took %.1fs, %.0f sentences/s, %.6f" %
			(word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0, total_error[0]))

		return (word_count[0], total_error[0])

	def create_job(self, sentence, vocab):
		return (vocab[" ".join(sentence)],[self.get_underlying_word_object(word) for word in sentence])

	def build_paragraph_vocab(self, sentences):
		paragraph_vocab = {}
		for sentence_no, sentence in enumerate(sentences):
			sentence = " ".join(sentence)
			if sentence not in paragraph_vocab:
				paragraph_vocab[sentence] = Paragraph()

		logger.info("collected %i sentence types from a corpus of %i sentences." %
				(len(paragraph_vocab), sentence_no+1))

		# assign a unique index to each sentence
		self.paragraph_vocab, self.index2paragraph = {}, []
		append = self.index2paragraph.append
		assign_to_vocab = self.paragraph_vocab.__setitem__

		for sentence, v in iteritems(paragraph_vocab):
			v.index = len(self.paragraph_vocab)
			assign_to_vocab(sentence, v)
			append(sentence)

	def fit(self, sentences, alpha = 0.035, max_iter = 100, max_batchsize = 500, verbose = True):
		"""
		Train using SGD with varying batchsizes the PV-DM machine to learn
		paragraph representations.

		Inputs
		------

		sentences: the corpus to learn from.
		max_iter: maximum number of training epochs to run for
		max_batchsize: maximum size of the training batch
		verbose: display current optimization error and epoch during training.

		Outputs
		-------

		list of errors at each epoch.
		"""
		errors = []
		max_batchsize = min(MAX_BATCHSIZE, max_batchsize)
		batch_size_change = False
		for i in range(max_iter):
			self.alpha = alpha * (1 - 1.0 * i / (max_iter-1))
			words, error = self.train(sentences)
			if verbose: print("Epoch %d: error %.3f, alpha = %.5f" % (i, error, self.alpha))
			if i > 0 and error > errors[-1] and not batch_size_change:
				errors.append(error)
				batch_size_change = False
				if self.batchsize == max_batchsize:
					break
				else:
					self.batchsize = min(max_batchsize, self.batchsize + 50)
					if verbose: print("==> Increasing batch size to %d " % (self.batchsize))
					batch_size_change = True
			else:
				errors.append(error)
				batch_size_change = False
		return errors

	def build_vocab(self, sentences, oov_word = False, report_frequency = 10000):
		"""
		Build vocabulary from a sequence of sentences (can be a once-only generator stream).
		Each sentence must be a list of utf8 strings.

		"""
		path = (re.sub("/","_",sentences.fname)+".vocab") if hasattr(sentences, "fname") else None
		if path != None and file_exists(path):
			logger.info("loading from saved vocab list at \"%s\"" % (path))
			file = gzip.open(path, 'r')
			saved_vocab = pickle.load(file)
			file.close()
			self.index2word = saved_vocab["index2word"]
			self.vocab      = saved_vocab["vocab"]

			if oov_word:
				self.add_oov_word(count = 100000)
			
			if PaddingWord not in self.vocab:
				v = self.add_word_to_vocab(PaddingWord, count = 1000000)
				self.padding_word = v
			else:
				self.padding_word = self.vocab[PaddingWord]

			# add special padding word here.
			self.create_binary_tree()
			self.build_paragraph_vocab(sentences)
			self.reset_weights()

		else:
			logger.info("collecting all words and their counts")

			prev_sentence_no = -1
			sentence_no, vocab = -1, {}
			total_words = 0
			assign_to_vocab = vocab.__setitem__ # slight performance gain
			# https://wiki.python.org/moin/PythonSpeed/PerformanceTips
			get_from_vocab = vocab.__getitem__
			for sentence_no, sentence in enumerate(sentences):
				if prev_sentence_no == sentence_no:
					break
				if sentence_no % report_frequency == 0:
					logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
						(sentence_no, total_words, len(vocab)))
				for word in sentence:
					if word in vocab:
						get_from_vocab(word).count += 1
					else:
						assign_to_vocab(word, Vocab(count=1))
				total_words += len(sentence)
				prev_sentence_no = sentence_no
			logger.info("collected %i word types from a corpus of %i words and %i sentences" %
				(len(vocab), total_words, sentence_no + 1))

			# assign a unique index to each word
			self.vocab, self.index2word = {}, []
			append = self.index2word.append
			assign_to_vocab = self.vocab.__setitem__
			for word, v in iteritems(vocab):
				if v.count >= self.min_count:
					v.index = len(self.vocab)
					append(word)
					assign_to_vocab(word, v)

			# add the special out of vocabulary word **UNKNOWN**:
			if oov_word: self.add_oov_word(count = len(vocab) - len(self.vocab))

			if PaddingWord not in self.vocab:
				v = self.add_word_to_vocab(PaddingWord, count = 1000000)
				self.padding_word = v
			else:
				self.padding_word = self.vocab[PaddingWord]

			logger.info("total %i word types after removing those with count<%s" % (len(self.vocab), self.min_count))

			# add info about each word's Huffman encoding
			self.create_binary_tree()
			self.build_paragraph_vocab(sentences)
			self.reset_weights()
			if path != None:
				logger.info("saving vocab list in \"%s\"" % (path))
				with gzip.open(path, 'wb') as file:
					pickle.dump({"vocab": self.vocab, "index2word": self.index2word}, file, 1)