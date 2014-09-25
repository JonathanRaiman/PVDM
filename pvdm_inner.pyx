#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.string cimport memset

from cpython cimport PyCapsule_GetPointer # PyCObject_AsVoidPtr
from scipy.linalg.blas import fblas

REAL = np.float32
ctypedef np.float32_t REAL_t
ctypedef np.uint32_t  INT_t

DEF MAX_SENTENCE_LEN = 100
DEF MAX_BATCH_SIZE   = 1000

ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil

cdef saxpy_ptr saxpy=<saxpy_ptr>PyCapsule_GetPointer(fblas.saxpy._cpointer , NULL)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCapsule_GetPointer(fblas.sdot._cpointer     , NULL)      # float = dot(x, y)

cdef sscal_ptr sscal=<sscal_ptr>PyCapsule_GetPointer(fblas.sscal._cpointer, NULL) # x = alpha * x

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

"""
PVDM concatenation, with multiple sentences at once.

TODO:
-Add option to use sum of vectors instead of concatenation,
-Add option to sample a single window for a paragraph.

"""

cdef void fast_sentence_sg_pvdm_batch_concatenation_skipgram(
    const np.uint32_t * points[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    const np.uint8_t *  codes[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    const int           codelens[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    REAL_t *            syn0, \
    REAL_t *            syn1, \
    REAL_t *            synparagraph, \
    const np.uint32_t   paragraph_index[MAX_BATCH_SIZE], \
    const int           word_size, \
    const int           paragraph_size, \
    const int           logistic_regression_size, \
    const np.uint32_t   indexes[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    const np.uint32_t   padding_word_index, \
    const REAL_t        alpha, \
    REAL_t *            paragraph_work, \
    REAL_t *            word_work, \
    REAL_t *            neu1, \
    int sentence_len[MAX_BATCH_SIZE], \
    int                 window, \
    bint                symmetric_window, \
    bint                paragraphs_only, \
    bint                scale_updates, \
    int                 batchsize, \
    REAL_t *            error,
    const np.uint32_t * random_windows) nogil:
    
    cdef long long a, b, batch
    cdef long long row2
    cdef REAL_t f, g, inv_count
    cdef int m, i, j, k, word_2_target
    cdef int pos
    cdef int window_size = word_size

    cdef np.uint32_t random_target
    
    for batch in range(batchsize):

        for i in range(sentence_len[batch]):

            if codelens[batch][i] == 0:
                continue

            random_target = random_windows[batch * MAX_SENTENCE_LEN + i]
            word_2_target = i + random_target

            if word_2_target == i:
                word_2_target += 1

            if word_2_target < 0 or word_2_target >= sentence_len[batch]:
                continue

            memset(word_work, 0, window_size * cython.sizeof(REAL_t))
            
            # 3. We reset neu1:
            memset(neu1,      0, logistic_regression_size * cython.sizeof(REAL_t))
            # 4. We load neu1 with the word to self predict:
            saxpy(&word_size, &ONEF, &syn0[indexes[batch][word_2_target] * word_size], &ONE, neu1, &ONE)

            for b in range(codelens[batch][i]):

                row2 = points[batch][i][b] * logistic_regression_size

                # multiply paragraph with regressor:
                f = <REAL_t>sdot(&logistic_regression_size, neu1, &ONE, &syn1[row2], &ONE)

                if f <= -MAX_EXP or f >= MAX_EXP:
                    # the output is -1 or 1, so no learning can be done:
                    continue

                # the activation for this branch in the hierarchical softmax tree:
                f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

                # the gradient for this binary code:
                g = (1 - codes[batch][i][b] - f) * alpha

                # Save the error (the squared difference with the desired binary code)
                error[0] += (1 - codes[batch][i][b] - f)**2

                # modify the word work:
                saxpy(&window_size, &g, &syn1[row2], &ONE, word_work, &ONE)
                
                # modify the regressor for this code:
                saxpy(&logistic_regression_size, &g, neu1, &ONE, &syn1[row2], &ONE)

            # After each window training, update the words
            # using the word work vector
            # (that has size = training window)
            saxpy(&word_size, &ONEF, word_work, &ONE, &syn0[indexes[batch][word_2_target] * word_size], &ONE)

cdef void fast_sentence_sg_pvdm_batch_concatenation_self_predict(
    const np.uint32_t * points[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    const np.uint8_t *  codes[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    const int           codelens[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    REAL_t *            syn0, \
    REAL_t *            syn1, \
    REAL_t *            synparagraph, \
    const np.uint32_t   paragraph_index[MAX_BATCH_SIZE], \
    const int           word_size, \
    const int           paragraph_size, \
    const int           logistic_regression_size, \
    const np.uint32_t   indexes[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    const np.uint32_t   padding_word_index, \
    const REAL_t        alpha, \
    REAL_t *            paragraph_work, \
    REAL_t *            word_work, \
    REAL_t *            neu1, \
    int sentence_len[MAX_BATCH_SIZE], \
    int                 window, \
    bint                symmetric_window, \
    bint                paragraphs_only, \
    bint                scale_updates, \
    int                 batchsize, \
    REAL_t *            error) nogil:
    
    cdef long long a, b, batch
    cdef long long row2
    cdef REAL_t f, g, inv_count
    cdef int m, i, j, k
    cdef int pos
    cdef int window_size = word_size
    
    for batch in range(batchsize):

        for i in range(sentence_len[batch]):

            if codelens[batch][i] == 0:
                continue

            memset(word_work, 0, window_size * cython.sizeof(REAL_t))
            
            # 3. We reset neu1:
            memset(neu1,      0, logistic_regression_size * cython.sizeof(REAL_t))
            # 4. We load neu1 with the word to self predict:
            saxpy(&word_size, &ONEF, &syn0[indexes[batch][i] * word_size], &ONE, neu1, &ONE)

            for b in range(codelens[batch][i]):

                row2 = points[batch][i][b] * logistic_regression_size

                # multiply paragraph with regressor:
                f = <REAL_t>sdot(&logistic_regression_size, neu1, &ONE, &syn1[row2], &ONE)

                if f <= -MAX_EXP or f >= MAX_EXP:
                    # the output is -1 or 1, so no learning can be done:
                    continue

                # the activation for this branch in the hierarchical softmax tree:
                f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

                # the gradient for this binary code:
                g = (1 - codes[batch][i][b] - f) * alpha

                # Save the error (the squared difference with the desired binary code)
                error[0] += (1 - codes[batch][i][b] - f)**2

                # modify the word work:
                saxpy(&window_size, &g, &syn1[row2], &ONE, word_work, &ONE)
                
                # modify the regressor for this code:
                saxpy(&logistic_regression_size, &g, neu1, &ONE, &syn1[row2], &ONE)

            # After each window training, update the words
            # using the word work vector
            # (that has size = training window)
            saxpy(&word_size, &ONEF, word_work, &ONE, &syn0[indexes[batch][i] * word_size], &ONE)

cdef void fast_sentence_sg_pvdm_batch_concatenation_rand_window(
    const np.uint32_t * points[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    const np.uint8_t *  codes[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    const int           codelens[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    REAL_t *            syn0, \
    REAL_t *            syn1, \
    REAL_t *            synparagraph, \
    const np.uint32_t   paragraph_index[MAX_BATCH_SIZE], \
    const int           word_size, \
    const int           paragraph_size, \
    const int           logistic_regression_size, \
    const np.uint32_t   indexes[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    const np.uint32_t   padding_word_index, \
    const REAL_t        alpha, \
    REAL_t *            paragraph_work, \
    REAL_t *            word_work, \
    REAL_t *            neu1, \
    int sentence_len[MAX_BATCH_SIZE], \
    int chosen_window[MAX_BATCH_SIZE], \
    int                 window, \
    bint                symmetric_window, \
    bint                paragraphs_only, \
    bint                scale_updates, \
    int                 batchsize, \
    REAL_t *            error) nogil:
    
    cdef long long a, b, batch
    cdef long long row2
    cdef REAL_t f, g, inv_count
    cdef int m, i, j, k
    cdef int pos
    cdef int window_size = (logistic_regression_size - paragraph_size)
    
    # Resets
    # ------
    #
    
    for batch in range(batchsize):
        # 1. We reset the update vector for the embedding of each batch's paragraph
        memset(paragraph_work, 0, paragraph_size * cython.sizeof(REAL_t))

        i = chosen_window[batch]

        if codelens[batch][i] == 0:
            continue
        j = i - window

        if symmetric_window:
            # here the word window is +/- window around center word
            # (predict center word only using past and future)
            k = i + window + 1
        else:
            # here the word window is just 'window' before target word
            # (predict next word only using previous)
            k = i

        # 2. We reset the update vector for each of the words in each of the batch's windows:
        if not paragraphs_only:
            memset(word_work, 0, window_size * cython.sizeof(REAL_t))
        
        # 3. We reset neu1:
        memset(neu1, 0, logistic_regression_size * cython.sizeof(REAL_t))
        # 4. We load neu1 with the paragraph, and the word window.
        pos = 0
        saxpy(&paragraph_size, &ONEF, &synparagraph[paragraph_index[batch] * paragraph_size], &ONE, neu1, &ONE)
        for m in range(j,k):
            if m == i:
                continue
            else:
                if m < 0 or m >= sentence_len[batch]:
                    # the word is the special **NULL** word, the padding word:
                    # ``If the paragraph has less than 9 words, we pre-pad with a special NULL word symbol."
                    saxpy(&word_size, &ONEF, &syn0[padding_word_index * word_size], &ONE, &neu1[paragraph_size + (pos * word_size)], &ONE)
                else:
                    # we load the normal word into neu1, at a position after the paragraph:
                    saxpy(&word_size, &ONEF, &syn0[indexes[batch][m] * word_size],  &ONE, &neu1[paragraph_size + (pos * word_size)], &ONE)
                pos += 1

        for b in range(codelens[batch][i]):

            row2 = points[batch][i][b] * logistic_regression_size

            # multiply paragraph with regressor:
            f = <REAL_t>sdot(&logistic_regression_size, neu1, &ONE, &syn1[row2], &ONE)

            if f <= -MAX_EXP or f >= MAX_EXP:
                # the output is -1 or 1, so no learning can be done:
                continue

            # the activation for this branch in the hierarchical softmax tree:
            f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

            # the gradient for this binary code:
            g = (1 - codes[batch][i][b] - f) * alpha

            # Save the error (the squared difference with the desired binary code)
            error[0] += (1 - codes[batch][i][b] - f)**2
            
            # modify the paragraph work work:
            saxpy(&paragraph_size, &g, &syn1[row2],                &ONE, paragraph_work, &ONE)

            if not paragraphs_only:
                # modify the word work:
                saxpy(&window_size,    &g, &syn1[row2+paragraph_size], &ONE, word_work, &ONE)
                
                # modify the regressor for this code:
                saxpy(&logistic_regression_size, &g, neu1, &ONE, &syn1[row2], &ONE)

        # After each window training, update the words
        # using the word work vector
        # (that has size = training window)
        if not paragraphs_only:
            pos = 0
            for m in range(j, k):
                if m == i:
                    # we didn't use the center word to make a prediction
                    continue
                else:
                    if m < 0 or m >= sentence_len[batch]:
                        # update the special padding word in syn0:
                        saxpy(&word_size, &ONEF, &word_work[pos * word_size], &ONE, &syn0[padding_word_index * word_size], &ONE)
                    else:
                        # update a word in syn0:
                        saxpy(&word_size, &ONEF, &word_work[pos * word_size], &ONE, &syn0[indexes[batch][m] * word_size], &ONE)
                    pos += 1 

        # After each sentence training, update the paragraph
        # using the paragraph work vector:

        # If number of updates to paragraph work > 1, the take average:
        if scale_updates and sentence_len[batch] > <int>1:
            inv_count = ONEF/sentence_len[batch]
            sscal(&paragraph_size, &inv_count, paragraph_work, &ONE)

        saxpy(&paragraph_size, &ONEF, paragraph_work, &ONE, &synparagraph[paragraph_index[batch] * paragraph_size], &ONE)

cdef void fast_sentence_sg_pvdm_batch_concatenation(
    const np.uint32_t * points[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    const np.uint8_t *  codes[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    const int           codelens[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    REAL_t *            syn0, \
    REAL_t *            syn1, \
    REAL_t *            synparagraph, \
    const np.uint32_t   paragraph_index[MAX_BATCH_SIZE], \
    const int           word_size, \
    const int           paragraph_size, \
    const int           logistic_regression_size, \
    const np.uint32_t   indexes[MAX_BATCH_SIZE][MAX_SENTENCE_LEN], \
    const np.uint32_t   padding_word_index, \
    const REAL_t        alpha, \
    REAL_t *            paragraph_work, \
    REAL_t *            word_work, \
    REAL_t *            neu1, \
    int sentence_len[MAX_BATCH_SIZE], \
    int                 window, \
    bint                symmetric_window, \
    bint                paragraphs_only, \
    bint                scale_updates, \
    int                 batchsize, \
    REAL_t *            error) nogil:
    
    cdef long long a, b, batch
    cdef long long row2
    cdef REAL_t f, g, inv_count
    cdef int m, i, j, k
    cdef int pos
    cdef int window_size = (logistic_regression_size - paragraph_size)
    
    # Resets
    # ------
    #
    
    for batch in range(batchsize):
        # 1. We reset the update vector for the embedding of each batch's paragraph
        memset(paragraph_work, 0, paragraph_size * cython.sizeof(REAL_t))

        for i in range(sentence_len[batch]):

            if codelens[batch][i] == 0:
                continue
            j = i - window
            if symmetric_window:
                # here the word window is +/- window around center word
                # (predict center word only using past and future)
                k = i + window + 1
            else:
                # here the word window is just 'window' before target word
                # (predict next word only using previous)
                k = i

            # 2. We reset the update vector for each of the words in each of the batch's windows:
            if not paragraphs_only:
                memset(word_work, 0, window_size * cython.sizeof(REAL_t))
            
            # 3. We reset neu1:
            memset(neu1, 0, logistic_regression_size * cython.sizeof(REAL_t))
            # 4. We load neu1 with the paragraph, and the word window.
            pos = 0
            saxpy(&paragraph_size, &ONEF, &synparagraph[paragraph_index[batch] * paragraph_size], &ONE, neu1, &ONE)
            for m in range(j,k):
                if m == i:
                    continue
                else:
                    if m < 0 or m >= sentence_len[batch]:
                        # the word is the special **NULL** word, the padding word:
                        # ``If the paragraph has less than 9 words, we pre-pad with a special NULL word symbol."
                        saxpy(&word_size, &ONEF, &syn0[padding_word_index * word_size], &ONE, &neu1[paragraph_size + (pos * word_size)], &ONE)
                    else:
                        # we load the normal word into neu1, at a position after the paragraph:
                        saxpy(&word_size, &ONEF, &syn0[indexes[batch][m] * word_size],  &ONE, &neu1[paragraph_size + (pos * word_size)], &ONE)
                    pos += 1

            for b in range(codelens[batch][i]):

                row2 = points[batch][i][b] * logistic_regression_size

                # multiply paragraph with regressor:
                f = <REAL_t>sdot(&logistic_regression_size, neu1, &ONE, &syn1[row2], &ONE)

                if f <= -MAX_EXP or f >= MAX_EXP:
                    # the output is -1 or 1, so no learning can be done:
                    continue

                # the activation for this branch in the hierarchical softmax tree:
                f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

                # the gradient for this binary code:
                g = (1 - codes[batch][i][b] - f) * alpha

                # Save the error (the squared difference with the desired binary code)
                error[0] += (1 - codes[batch][i][b] - f)**2
                
                # modify the paragraph work work:
                saxpy(&paragraph_size, &g, &syn1[row2],                &ONE, paragraph_work, &ONE)

                if not paragraphs_only:
                    # modify the word work:
                    saxpy(&window_size,    &g, &syn1[row2+paragraph_size], &ONE, word_work, &ONE)
                    
                    # modify the regressor for this code:
                    saxpy(&logistic_regression_size, &g, neu1, &ONE, &syn1[row2], &ONE)

            # After each window training, update the words
            # using the word work vector
            # (that has size = training window)
            if not paragraphs_only:
                pos = 0
                for m in range(j, k):
                    if m == i:
                        # we didn't use the center word to make a prediction
                        continue
                    else:
                        if m < 0 or m >= sentence_len[batch]:
                            # update the special padding word in syn0:
                            saxpy(&word_size, &ONEF, &word_work[pos * word_size], &ONE, &syn0[padding_word_index * word_size], &ONE)
                        else:
                            # update a word in syn0:
                            saxpy(&word_size, &ONEF, &word_work[pos * word_size], &ONE, &syn0[indexes[batch][m] * word_size], &ONE)
                        pos += 1 

        # After each sentence training, update the paragraph
        # using the paragraph work vector:

        # If number of updates to paragraph work > 1, the take average:
        if scale_updates and sentence_len[batch] > <int>1:
            inv_count = ONEF/sentence_len[batch]
            sscal(&paragraph_size, &inv_count, paragraph_work, &ONE)

        saxpy(&paragraph_size, &ONEF, paragraph_work, &ONE, &synparagraph[paragraph_index[batch] * paragraph_size], &ONE)

def train_sentence_batch_pvdm_skipgram(model, sentences, paragraphs, _paragraphs_only, _alpha, _paragraph_work, _word_work, _neu1, _error, int batchsize):
    cdef REAL_t *syn0         = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *syn1         = <REAL_t *>(np.PyArray_DATA(model.syn1))
    cdef REAL_t *synparagraph = <REAL_t *>(np.PyArray_DATA(paragraphs))
    cdef REAL_t *error        = <REAL_t *>(np.PyArray_DATA(_error))

    cdef np.uint32_t padding_word_index = model.padding_word.index
    cdef long long batch

    cdef REAL_t *paragraph_work
    cdef REAL_t *word_work
    cdef REAL_t *neu1
    cdef np.uint32_t word2_index
    cdef REAL_t alpha = _alpha

    # sizes: (for saxpy and sdot)
    cdef int word_size                = model.layer1_size
    cdef int paragraph_size           = model.paragraph_size
    cdef int logistic_regression_size = model.logistic_regression_size
    cdef int window                   = model.window


    _random_windows = np.random.randint(-window, window, (MAX_BATCH_SIZE * MAX_SENTENCE_LEN)).astype('uint32')

    cdef np.uint32_t * random_windows = <np.uint32_t *>(np.PyArray_DATA(_random_windows))

    cdef np.uint32_t   paragraph_index[MAX_BATCH_SIZE]
    cdef np.uint32_t * points[MAX_BATCH_SIZE][MAX_SENTENCE_LEN]
    cdef np.uint8_t *  codes[MAX_BATCH_SIZE][MAX_SENTENCE_LEN]
    cdef int           codelens[MAX_BATCH_SIZE][MAX_SENTENCE_LEN]
    cdef np.uint32_t   indexes[MAX_BATCH_SIZE][MAX_SENTENCE_LEN]
    cdef int           sentence_len[MAX_BATCH_SIZE]
    
    cdef long words_seen = 0

    cdef bint paragraphs_only  = <bint>_paragraphs_only
    cdef bint concatenate      = <bint>model.concatenate
    cdef bint symmetric_window = <bint>model.symmetric_window
    cdef bint scale_updates    = <bint>model.scale_updates

    # convert Python structures to primitive types, so we can release the GIL
    paragraph_work = <REAL_t *>np.PyArray_DATA(_paragraph_work) # space to work on paragraph
    neu1           = <REAL_t *>np.PyArray_DATA(_neu1)           # space for input to syn1
    if not paragraphs_only:
        word_work      = <REAL_t *>np.PyArray_DATA(_word_work)      # space to work on words.

    for batch in range(batchsize):
        paragraph_index[batch] = sentences[batch][0].index
        sentence_len[batch]    = <int>min(MAX_SENTENCE_LEN, len(sentences[batch][1]))

        for i in range(sentence_len[batch]):
            word = sentences[batch][1][i]
            if word is None:
                codelens[batch][i] = 0
            else:
                indexes[batch][i]  = word.index
                codelens[batch][i] = <int>len(word.code)
                points[batch][i]   = <np.uint32_t *> np.PyArray_DATA(word.point)
                codes[batch][i]    = <np.uint8_t *>  np.PyArray_DATA(word.code)
                words_seen += 1

    # release GIL & train on the sentence
    with nogil:
        if concatenate:
            fast_sentence_sg_pvdm_batch_concatenation_skipgram(
                points, \
                codes, \
                codelens, \
                syn0, \
                syn1, \
                synparagraph, \
                paragraph_index, \
                word_size, \
                paragraph_size, \
                logistic_regression_size, \
                indexes, \
                padding_word_index, \
                alpha, \
                paragraph_work, \
                word_work, \
                neu1, \
                sentence_len, \
                window, \
                symmetric_window, \
                paragraphs_only, \
                scale_updates, \
                batchsize, \
                error, \
                random_windows)

    return words_seen

def train_sentence_batch_pvdm_self_predict(model, sentences, paragraphs, _paragraphs_only, _alpha, _paragraph_work, _word_work, _neu1, _error, int batchsize):
    cdef REAL_t *syn0         = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *syn1         = <REAL_t *>(np.PyArray_DATA(model.syn1))
    cdef REAL_t *synparagraph = <REAL_t *>(np.PyArray_DATA(paragraphs))
    cdef REAL_t *error        = <REAL_t *>(np.PyArray_DATA(_error))

    cdef np.uint32_t padding_word_index = model.padding_word.index
    cdef long long batch

    cdef REAL_t *paragraph_work
    cdef REAL_t *word_work
    cdef REAL_t *neu1
    cdef np.uint32_t word2_index
    cdef REAL_t alpha = _alpha

    # sizes: (for saxpy and sdot)
    cdef int word_size                = model.layer1_size
    cdef int paragraph_size           = model.paragraph_size
    cdef int logistic_regression_size = model.logistic_regression_size
    cdef int window                   = model.window

    cdef np.uint32_t   paragraph_index[MAX_BATCH_SIZE]
    cdef np.uint32_t * points[MAX_BATCH_SIZE][MAX_SENTENCE_LEN]
    cdef np.uint8_t *  codes[MAX_BATCH_SIZE][MAX_SENTENCE_LEN]
    cdef int           codelens[MAX_BATCH_SIZE][MAX_SENTENCE_LEN]
    cdef np.uint32_t   indexes[MAX_BATCH_SIZE][MAX_SENTENCE_LEN]
    cdef int           sentence_len[MAX_BATCH_SIZE]
    
    cdef long words_seen = 0

    cdef bint paragraphs_only  = <bint>_paragraphs_only
    cdef bint concatenate      = <bint>model.concatenate
    cdef bint symmetric_window = <bint>model.symmetric_window
    cdef bint scale_updates    = <bint>model.scale_updates

    # convert Python structures to primitive types, so we can release the GIL
    paragraph_work = <REAL_t *>np.PyArray_DATA(_paragraph_work) # space to work on paragraph
    neu1           = <REAL_t *>np.PyArray_DATA(_neu1)           # space for input to syn1
    if not paragraphs_only:
        word_work      = <REAL_t *>np.PyArray_DATA(_word_work)      # space to work on words.

    for batch in range(batchsize):
        paragraph_index[batch] = sentences[batch][0].index
        sentence_len[batch]    = <int>min(MAX_SENTENCE_LEN, len(sentences[batch][1]))

        for i in range(sentence_len[batch]):
            word = sentences[batch][1][i]
            if word is None:
                codelens[batch][i] = 0
            else:
                indexes[batch][i]  = word.index
                codelens[batch][i] = <int>len(word.code)
                points[batch][i]   = <np.uint32_t *> np.PyArray_DATA(word.point)
                codes[batch][i]    = <np.uint8_t *>  np.PyArray_DATA(word.code)
                words_seen += 1

    # release GIL & train on the sentence
    with nogil:
        if concatenate:
            fast_sentence_sg_pvdm_batch_concatenation_self_predict(
                points, \
                codes, \
                codelens, \
                syn0, \
                syn1, \
                synparagraph, \
                paragraph_index, \
                word_size, \
                paragraph_size, \
                logistic_regression_size, \
                indexes, \
                padding_word_index, \
                alpha, \
                paragraph_work, \
                word_work, \
                neu1, \
                sentence_len, \
                window, \
                symmetric_window, \
                paragraphs_only, \
                scale_updates, \
                batchsize, \
                error)

    return words_seen

def train_sentence_batch_pvdm(model, sentences, paragraphs, _paragraphs_only, _alpha, _paragraph_work, _word_work, _neu1, _error, int batchsize):
    cdef REAL_t *syn0         = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *syn1         = <REAL_t *>(np.PyArray_DATA(model.syn1))
    cdef REAL_t *synparagraph = <REAL_t *>(np.PyArray_DATA(paragraphs))
    cdef REAL_t *error        = <REAL_t *>(np.PyArray_DATA(_error))

    cdef np.uint32_t padding_word_index = model.padding_word.index
    cdef long long batch

    cdef REAL_t *paragraph_work
    cdef REAL_t *word_work
    cdef REAL_t *neu1
    cdef np.uint32_t word2_index
    cdef REAL_t alpha = _alpha

    # sizes: (for saxpy and sdot)
    cdef int word_size                = model.layer1_size
    cdef int paragraph_size           = model.paragraph_size
    cdef int logistic_regression_size = model.logistic_regression_size
    cdef int window                   = model.window

    cdef np.uint32_t   paragraph_index[MAX_BATCH_SIZE]
    cdef np.uint32_t * points[MAX_BATCH_SIZE][MAX_SENTENCE_LEN]
    cdef np.uint8_t *  codes[MAX_BATCH_SIZE][MAX_SENTENCE_LEN]
    cdef int           codelens[MAX_BATCH_SIZE][MAX_SENTENCE_LEN]
    cdef np.uint32_t   indexes[MAX_BATCH_SIZE][MAX_SENTENCE_LEN]
    cdef int           sentence_len[MAX_BATCH_SIZE]
    cdef int random_window[MAX_BATCH_SIZE]
    
    
    cdef long words_seen = 0

    cdef bint paragraphs_only  = <bint>_paragraphs_only
    cdef bint concatenate      = <bint>model.concatenate
    cdef bint symmetric_window = <bint>model.symmetric_window
    cdef bint scale_updates    = <bint>model.scale_updates
    cdef bint learn_on_random_windows = <bint>model.random_window

    # convert Python structures to primitive types, so we can release the GIL
    paragraph_work = <REAL_t *>np.PyArray_DATA(_paragraph_work) # space to work on paragraph
    neu1           = <REAL_t *>np.PyArray_DATA(_neu1)           # space for input to syn1
    if not paragraphs_only:
        word_work      = <REAL_t *>np.PyArray_DATA(_word_work)      # space to work on words.

    for batch in range(batchsize):
        paragraph_index[batch] = sentences[batch][0].index
        sentence_len[batch]    = <int>min(MAX_SENTENCE_LEN, len(sentences[batch][1]))
        if learn_on_random_windows:
            random_window[batch] = np.random.randint(sentence_len[batch])

        for i in range(sentence_len[batch]):
            word = sentences[batch][1][i]
            if word is None:
                codelens[batch][i] = 0
            else:
                indexes[batch][i]  = word.index
                codelens[batch][i] = <int>len(word.code)
                points[batch][i]   = <np.uint32_t *> np.PyArray_DATA(word.point)
                codes[batch][i]    = <np.uint8_t *>  np.PyArray_DATA(word.code)
                words_seen += 1

    # release GIL & train on the sentence
    with nogil:
        # not concatenated is not yet implemented.
        if concatenate:
            if learn_on_random_windows:
                fast_sentence_sg_pvdm_batch_concatenation_rand_window(
                    points, \
                    codes, \
                    codelens, \
                    syn0, \
                    syn1, \
                    synparagraph, \
                    paragraph_index, \
                    word_size, \
                    paragraph_size, \
                    logistic_regression_size, \
                    indexes, \
                    padding_word_index, \
                    alpha, \
                    paragraph_work, \
                    word_work, \
                    neu1, \
                    sentence_len, \
                    random_window, \
                    window, \
                    symmetric_window, \
                    paragraphs_only, \
                    scale_updates, \
                    batchsize, \
                    error)
            else:
                fast_sentence_sg_pvdm_batch_concatenation(
                    points, \
                    codes, \
                    codelens, \
                    syn0, \
                    syn1, \
                    synparagraph, \
                    paragraph_index, \
                    word_size, \
                    paragraph_size, \
                    logistic_regression_size, \
                    indexes, \
                    padding_word_index, \
                    alpha, \
                    paragraph_work, \
                    word_work, \
                    neu1, \
                    sentence_len, \
                    window, \
                    symmetric_window, \
                    paragraphs_only, \
                    scale_updates, \
                    batchsize, \
                    error)

    return words_seen

cdef void fast_sentence_sg_pvdm_concatenation(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, REAL_t *synparagraph, const np.uint32_t paragraph_index, \
    const int word_size, const int paragraph_size, const int logistic_regression_size,
    const np.uint32_t *indexes, const np.uint32_t padding_word_index, const REAL_t alpha, \
    REAL_t *paragraph_work, REAL_t *word_work, REAL_t *neu1,\
    int i, int j, int k, int sentence_len, REAL_t *error) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g
    cdef int m
    cdef int pos
    cdef int window_size = (logistic_regression_size - paragraph_size)

    # Resets
    # ------
    #
    # 1. We reset the update vector for the embedding of a paragraph
    memset(paragraph_work, 0, paragraph_size * cython.sizeof(REAL_t))

    # 2. We reset the update vector for the word:
    memset(word_work, 0, window_size * cython.sizeof(REAL_t))

    # 3. We reset neu1:
    memset(neu1, 0, logistic_regression_size * cython.sizeof(REAL_t))
    pos = 0
    saxpy(&paragraph_size, &ONEF, &synparagraph[paragraph_index * paragraph_size], &ONE, &neu1[paragraph_size + pos * word_size], &ONE)
    for m in range(j,k):
        if m == i:
            continue
        else:
            if m < 0 or m >= sentence_len:
                saxpy(&word_size, &ONEF, &syn0[padding_word_index * word_size], &ONE, &neu1[paragraph_size + pos * word_size], &ONE)
            else:
                saxpy(&word_size, &ONEF, &syn0[indexes[m] * word_size], &ONE, &neu1[paragraph_size + pos * word_size], &ONE)
            pos += 1

    for b in range(codelen):
        row2 = word_point[b] * logistic_regression_size

        # multiply paragraph with regressor:
        f = <REAL_t>sdot(&logistic_regression_size, neu1, &ONE, &syn1[row2], &ONE)

        if f <= -MAX_EXP or f >= MAX_EXP:
            continue

        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]        

        g = (1 - word_code[b] - f) * alpha
        
        error[0] += (1 - word_code[b] - f)**2

        # modify the paragraph work work:
        saxpy(&paragraph_size, &g, &syn1[row2], &ONE, paragraph_work, &ONE)

        # modify the word work:
        saxpy(&window_size, &g, &syn1[row2+paragraph_size], &ONE, word_work, &ONE)

        # modify the regressor for this code:
        saxpy(&logistic_regression_size, &g, neu1, &ONE, &syn1[row2], &ONE)
    
    # update paragraph:
    saxpy(&paragraph_size, &ONEF, paragraph_work, &ONE, &synparagraph[paragraph_index * paragraph_size], &ONE)
    # update word:
    pos = 0
    for m in range(j, k):
        if m == i:
            continue
        else:
            # here we should check that j and k are within
            # the sentence
            if m < 0 or m >= sentence_len:
                saxpy(&word_size, &ONEF, &word_work[pos * word_size], &ONE, &syn0[padding_word_index*word_size], &ONE)
            else:
                saxpy(&word_size, &ONEF, &word_work[pos * word_size], &ONE, &syn0[indexes[m]*word_size], &ONE)
            pos += 1


cdef void fast_sentence_sg_pvdm_summation(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, REAL_t *synparagraph, const np.uint32_t paragraph_index, \
    const int word_size, const int paragraph_size, const int logistic_regression_size,
    const np.uint32_t *indexes, const np.uint32_t padding_word_index, const REAL_t alpha, \
    REAL_t *paragraph_work, REAL_t *word_work, REAL_t *neu1,\
    int i, int j, int k, int sentence_len, REAL_t *error) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count
    cdef int m

    # Resets
    # ------
    #
    # 1. We reset the update vector for the embedding of a paragraph
    memset(paragraph_work, 0, paragraph_size * cython.sizeof(REAL_t))

    # 2. We reset the update vector for the word:
    memset(word_work, 0, word_size * cython.sizeof(REAL_t))

    # 3. We reset neu1 to store the average of the word vectors:
    memset(neu1, 0, word_size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            if m < 0 or m >= sentence_len:
                saxpy(&word_size, &ONEF, &syn0[padding_word_index * word_size], &ONE, neu1, &ONE)
            else:
                saxpy(&word_size, &ONEF, &syn0[indexes[m] * word_size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
        sscal(&word_size, &inv_count , neu1, &ONE)

    for b in range(codelen):
        row2 = word_point[b] * logistic_regression_size

        # multiply paragraph with regressor:
        f = <REAL_t>sdot(&paragraph_size, &synparagraph[paragraph_index * paragraph_size], &ONE, &syn1[row2], &ONE)
        # multiply word with regressor:
        f += <REAL_t>sdot(&word_size, neu1, &ONE, &syn1[row2+paragraph_size], &ONE)

        if f <= -MAX_EXP or f >= MAX_EXP:
            continue

        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]        

        g = (1 - word_code[b] - f) * alpha
        error[0] += (1 - word_code[b] - f)**2

        # modify the paragraph work work:
        saxpy(&paragraph_size, &g, &syn1[row2], &ONE, paragraph_work, &ONE)

        # modify the word work:
        saxpy(&word_size, &g, &syn1[row2+paragraph_size], &ONE, word_work, &ONE)

        # modify the regressor for this code:
        saxpy(&word_size, &g, neu1, &ONE, &syn1[row2+paragraph_size], &ONE)
        saxpy(&paragraph_size, &g, &synparagraph[paragraph_index * paragraph_size], &ONE, &syn1[row2], &ONE)
    
    # update paragraph:
    saxpy(&paragraph_size, &ONEF, paragraph_work, &ONE, &synparagraph[paragraph_index * paragraph_size], &ONE)
    # update words:
    for m in range(j, k):
        if m == i:
            continue
        else:
            if m < 0 or m >= sentence_len:
                saxpy(&word_size, &ONEF, word_work, &ONE, &syn0[padding_word_index*word_size], &ONE)
            else:
                saxpy(&word_size, &ONEF, word_work, &ONE, &syn0[indexes[m]*word_size], &ONE)

def train_sentence_pvdm(model, paragraph_object, sentence, alpha, _paragraph_work, _word_work, _neu1, _error):
    cdef REAL_t *syn0     = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *syn1     = <REAL_t *>(np.PyArray_DATA(model.syn1))
    cdef REAL_t *synparagraph = <REAL_t *>(np.PyArray_DATA(model.synparagraph))
    cdef REAL_t *error = <REAL_t *>(np.PyArray_DATA(_error))

    cdef np.uint32_t   padding_word_index  = model.padding_word.index
    cdef np.uint32_t   paragraph_index  = paragraph_object.index

    cdef REAL_t *paragraph_work
    cdef REAL_t *word_work
    cdef REAL_t *neu1
    cdef np.uint32_t word2_index
    cdef REAL_t _alpha = alpha

    # sizes: (for saxpy and sdot)
    cdef int word_size = model.layer1_size
    cdef int paragraph_size = model.paragraph_size
    cdef int logistic_regression_size = model.logistic_regression_size

    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]
    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    cdef bint concatenate = <bint>model.concatenate

    # convert Python structures to primitive types, so we can release the GIL
    paragraph_work = <REAL_t *>np.PyArray_DATA(_paragraph_work) # space to work on paragraph
    word_work      = <REAL_t *>np.PyArray_DATA(_word_work)      # space to work on words.
    neu1           = <REAL_t *>np.PyArray_DATA(_neu1)           # space for input to syn1
    sentence_len   = <int>min(MAX_SENTENCE_LEN, len(sentence))
    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            codelens[i] = <int>len(word.code)
            codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
            points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            result += 1

    # release GIL & train on the sentence
    with nogil:
        if not concatenate:
            for i in range(sentence_len):
                if codelens[i] == 0:
                    continue
                j = i - window
                k = i + window + 1
                fast_sentence_sg_pvdm_summation(points[i], codes[i], codelens[i],
                    syn0, syn1, synparagraph, paragraph_index,
                    word_size, paragraph_size, logistic_regression_size, \
                    indexes, padding_word_index, _alpha, paragraph_work, word_work, neu1, i, j, k, sentence_len, error)
        else:
            for i in range(sentence_len):
                if codelens[i] == 0:
                    continue
                j = i - window
                k = i + window + 1
                fast_sentence_sg_pvdm_concatenation(points[i], codes[i], codelens[i],
                    syn0, syn1, synparagraph, paragraph_index,
                    word_size, paragraph_size, logistic_regression_size, \
                    indexes, padding_word_index, _alpha, paragraph_work, word_work, neu1, i, j, k, sentence_len, error)

    return result

def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.

    """
    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))

    return 0

FAST_VERSION = init()  # initialize the module