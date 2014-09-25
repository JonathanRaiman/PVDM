### 


    from matplotlib import pyplot as plt
    from tsne import bh_sne
    from utils import LogisticRegression
    from sklearn.linear_model import LogisticRegression as skLogisticRegression
    from sklearn.metrics import classification_report
    from gensim import matutils
    import numpy as np, utils
    from pvdm import PVDM
    from word2vec_extended import LineCorpus
    from pvdm import PVDM, Paragraph
    from gensim.models.word2vec import Text8Corpus, logger
    trees = utils.import_tree_corpus("sentiment_data/other_data/train.txt")
    labels_hash = trees.labels()
    tree_export_path = "sentiment_data/all_lines.txt"
    tree_corpus = LineCorpus(tree_export_path, filter_lines = False)
    #tree_corpus = [tree.to_lines()[0].split() for tree in trees]
    pvdm = PVDM(concatenate = True, random_window = True, workers = 8, window = 8, batchsize = 1000, paragraph_size=400, decay = False, size = 400, alpha=0.035, symmetric_window=False, self_predict= False)
    pvdm.build_vocab(sentences= tree_corpus, oov_word = True)



    alpha = 0.035
    epochs = 130
    errors = []
    try:
        for epoch in range(epochs):
            pvdm.alpha = max(0.0001, alpha * (1 - 1.0 * epoch / (epochs-1)))
            words, error = pvdm.train(tree_corpus, paragraphs_only = False)
            print("Epoch %d: error %.3f, alpha = %.5f" % (epoch, error, pvdm.alpha))
            if error < 5:
                break
            if epoch > 0 and error > errors[-1]:
                errors.append(error)
            else:
                errors.append(error)
    except KeyboardInterrupt:
        print("Epoch %d: error %.3f, alpha = %.5f" % (epoch, error, pvdm.alpha))

    Epoch 0: error 12238564.014, alpha = 0.03500
    Epoch 1: error 9035877.274, alpha = 0.03473
    Epoch 2: error 7462139.332, alpha = 0.03446
    Epoch 3: error 6378156.541, alpha = 0.03419
    Epoch 4: error 5566759.061, alpha = 0.03391
    Epoch 5: error 4954506.754, alpha = 0.03364
    Epoch 6: error 4433725.362, alpha = 0.03337
    ...
    Epoch 127: error 907829.069, alpha = 0.00054
    Epoch 128: error 1058162.380, alpha = 0.00027
    Epoch 129: error 1215409.188, alpha = 0.00010



    #if in a self_predict mode
    pvdm.reintroduce_multi_word_windows()

    plt.plot(errors)
    plt.title("PV-DM: Bytes missed per training epoch"); plt.xlabel("training epoch"); plt.ylabel("Number of bytes");


![png](README_images/PV-DM%20-%20Batch%20Learning_5_0.png)



    # use this to verify work arrays are being written to
    jobs = (pvdm.create_job(sentence) for sentence in tree_corpus)
    paragraph_work = np.zeros(pvdm.paragraph_size, dtype=np.float32)  # each thread must have its own work memory
    word_work = np.zeros((pvdm.logistic_regression_size - pvdm.paragraph_size), dtype= np.float32)
    neu1 = matutils.zeros_aligned(pvdm.logistic_regression_size, dtype=np.float32)
    error = np.zeros(1, dtype = np.float32)
    job = [next(jobs)]

### 2D projections of Paragraphs


    full_lines = [tree.to_lines()[0] for tree in trees]

#### Using T-SNE to get 2D projection of paragraphs


    para_indices = [pvdm.paragraph_vocab[line].index for line in full_lines]
    X_2dpara = bh_sne(pvdm.synparagraph[para_indices].astype('float64'))

#### Visualize the projections


    min_visible = 10
    max_visible = 50
    para_labels = full_lines
    para_colors = [labels_hash[label] for label in para_labels[min_visible:min_visible+max_visible]]
    axes = plt.axes([0, 1, 3, 3])
    axes.scatter(X_2dpara[min_visible:min_visible+max_visible,0], X_2dpara[min_visible:min_visible+max_visible, 1], marker = 'o', cmap=plt.cm.seismic, s = 80)
    for label, x, y in zip(para_labels[min_visible:min_visible+max_visible], X_2dpara[min_visible:min_visible+max_visible, 0], X_2dpara[min_visible:min_visible+max_visible, 1]):
        plt.annotate(
            label, 
            xy = (x, y), xytext = (-8, -3),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = None,
            arrowprops = None)#dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


![png](README_images/PV-DM%20-%20Batch%20Learning_12_0.png)


Synparagraph using asymmetric window:


    startpoint = 4571
    plt.matshow(pvdm.synparagraph[startpoint:startpoint+1000, 0:400])
    plt.colorbar()




    <matplotlib.colorbar.Colorbar at 0x138e7e190>




![png](README_images/PV-DM%20-%20Batch%20Learning_14_1.png)



    startpoint = 4571
    plt.matshow(pvdm.synparagraph[startpoint:startpoint+1000, 0:400])
    plt.colorbar()




    <matplotlib.colorbar.Colorbar at 0x116339a50>




![png](README_images/PV-DM%20-%20Batch%20Learning_15_1.png)


Synparagraph using symmetric window:


    startpoint = 4571
    plt.matshow(pvdm.synparagraph[startpoint:startpoint+1000, 0:400])
    plt.colorbar()




    <matplotlib.colorbar.Colorbar at 0x13b53f7d0>




![png](README_images/PV-DM%20-%20Batch%20Learning_17_1.png)



    startpoint = 4671
    plt.matshow(pvdm.syn1[startpoint:startpoint+500, :400])
    plt.colorbar()
    plt.xlabel("Softmax Parameters")
    plt.ylabel("Tree branch index");


![png](README_images/PV-DM%20-%20Batch%20Learning_18_0.png)


After 30 several iterations we get:


    startpoint = 4671
    plt.matshow(pvdm.syn1[startpoint:startpoint+500, :400])
    plt.colorbar()
    plt.xlabel("Softmax Parameters")
    plt.ylabel("Tree branch index")




    <matplotlib.text.Text at 0x137534e10>




![png](README_images/PV-DM%20-%20Batch%20Learning_20_1.png)


After 60 several iterations we get:


    startpoint = 4671
    plt.matshow(pvdm.syn1[startpoint:startpoint+500, :400])
    plt.colorbar()
    plt.xlabel("Softmax Parameters")
    plt.ylabel("Tree branch index");


![png](README_images/PV-DM%20-%20Batch%20Learning_22_0.png)


# Sentiment Analysis

## Obtain labeled dataset


    labels = [labels_hash[label] for label in pvdm.index2paragraph]

## Train classifiers

Custom Logistic Regression classifier:


    reg = LogisticRegression(embedding_size=pvdm.layer1_size, target_size=5, weight_decay=True, alpha = 1.5, criterion = 'kl_divergence')
    errs = reg.sgd_fit(pvdm.synparagraph, labels, flush_output=True, verbose = True, max_iter = 400, batchsize=400)
    #reg.fit_until_convergence(pvdm.synparagraph[para_indices], labels, verbose =True, flush_output=True);

    Epoch 38: mean kl_divergence 1.249, 81795 / 159271 correct, alpha = 1.357, remaining tries 3



    reg.alpha.set_value(np.float32(0.2))
    errs = reg.sgd_fit(pvdm.synparagraph, labels, flush_output=True, verbose = True, max_iter = 400, batchsize=400)

    Epoch 59: mean kl_divergence 1.282, 81454 / 159271 correct, alpha = 0.170, remaining tries 3


Sk-learn Logistic Regression classifier: (too slow for full dataset)


    # too slow:
    #sk_reg = skLogisticRegression(tol=0.0000001)
    #sk_reg.fit(pvdm.synparagraph[para_indices], labels);

## Sentiment Classification Reports

### Classification report for Sk-Learn Logistic Regression


    print(classification_report(labels, reg.predict(pvdm.synparagraph), labels=[0,1,2,3,4], target_names=["--", "-", "", "+", "++"]))

                 precision    recall  f1-score   support
    
             --       0.07      0.00      0.00      7237
              -       0.24      0.01      0.01     27714
                      0.52      0.98      0.68     81638
              +       0.39      0.05      0.09     33361
             ++       0.21      0.01      0.01      9321
    
    avg / total       0.40      0.51      0.37    159271
    



    print(classification_report(labels, sk_reg.predict(pvdm.synparagraph), labels=[0,1,2,3,4], target_names=["--", "-", "", "+", "++"]))

                 precision    recall  f1-score   support
    
             --       0.30      0.02      0.04      1092
              -       0.32      0.46      0.38      2218
                      0.34      0.12      0.17      1624
              +       0.32      0.61      0.42      2322
             ++       0.37      0.08      0.13      1288
    
    avg / total       0.33      0.32      0.27      8544
    


### Classification report for custom Logistic Regression


    print(classification_report(labels, reg.predict(pvdm.synparagraph), labels=[0,1,2,3,4], target_names=["--", "-", "", "+", "++"]))

                 precision    recall  f1-score   support
    
             --       0.00      0.00      0.00      7237
              -       0.21      0.00      0.00     27714
                      0.51      0.99      0.68     81638
              +       0.31      0.01      0.02     33361
             ++       0.10      0.00      0.00      9321
    
    avg / total       0.37      0.51      0.35    159271
    


## Visualize the prediction confusion


    plt.matshow(reg.predict_proba(pvdm.synparagraph[20:60]), interpolation='nearest', cmap=plt.cm.Spectral)
    plt.colorbar();


![png](README_images/PV-DM%20-%20Batch%20Learning_39_0.png)



    plt.matshow(sk_reg.predict_proba(pvdm.synparagraph[0:20]), interpolation='nearest', cmap=plt.cm.Spectral)
    plt.colorbar();

## Compare labelings:


    list(reg.predict(pvdm.synparagraph[0:20]))




    [2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]




    labels[0:20]




    [3, 2, 2, 2, 4, 3, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]



## Get number of correct predictions


    np.equal(reg.predict(pvdm.synparagraph), np.array(labels)).sum()




    81438



## Qualitative Test

### Search for nearest paragraph


    pvdm.most_similar("darkness")




    [('lameness', 0.7936276793479919),
     ('anti-virus', 0.7769980430603027),
     ('contributions', 0.7643187642097473),
     ('preaches', 0.7543584704399109),
     ('pinnacle', 0.7475813031196594),
     ('multiplex', 0.7469998598098755),
     ('art-conscious', 0.7468372583389282),
     ('dictates', 0.7405220866203308),
     ('witch', 0.7355045080184937),
     ('owed', 0.7308557033538818)]




    for label, score in pvdm.most_similar_paragraph("bad"):
        print("%s - %.2f" % (label, score))
        #print("prediction %d vs actual %d" % (reg.predict([pvdm.synparagraph[pvdm.paragraph_vocab[label].index]])[0], labels_hash[label]))
        print()
        

    lot - 0.93
    prediction 0 vs actual 2
    
    down - 0.89
    prediction 2 vs actual 2
    
    car pileup - 0.88
    prediction 2 vs actual 2
    
    before - 0.87
    prediction 2 vs actual 2
    
    powerful - 0.84
    prediction 1 vs actual 4
    
    strong subject matter - 0.83
    prediction 2 vs actual 2
    
    bad sign - 0.82
    prediction 2 vs actual 1
    
    lot to be desired - 0.82
    prediction 2 vs actual 1
    
    bad bluescreen - 0.82
    prediction 2 vs actual 0
    
    bad boy weirdo role - 0.79
    prediction 2 vs actual 2
    


## Train new paragraphs:


    test_trees = utils.import_tree_corpus("sentiment_data/other_data/test.txt")
    index2paragraph_test = [tree.to_lines()[0] for tree in test_trees]
    test_paragraph_vocab = {}
    for line in index2paragraph_test:
        p = Paragraph()
        p.index = len(test_paragraph_vocab)
        test_paragraph_vocab[line]= p
    paragraphs = (np.random.randn(len(test_paragraph_vocab), pvdm.paragraph_size) * 1.0 / pvdm.paragraph_size) .astype(dtype=np.float32)

Re-optimize the paragraph vectors for training set:


    alpha = 0.035
    epochs = 300
    retraining_errors = []
    try:
        for epoch in range(epochs):
            pvdm.alpha = max(0.0001, alpha * (1 - 1.0 * epoch / (epochs-1)))
            words, error = pvdm.train(tree_corpus, paragraphs_only=True)
            print("Epoch %d: error %.3f, alpha = %.5f" % (epoch, error, pvdm.alpha))
            if error < 5:
                break
            if epoch > 0 and error > errors[-1]:
                retraining_errors.append(error)
            else:
                retraining_errors.append(error)
    except KeyboardInterrupt:
        print("Epoch %d: error %.3f, alpha = %.5f" % (epoch, error, pvdm.alpha))

    Epoch 0: error 259231.325, alpha = 0.03500
    Epoch 1: error 149707.214, alpha = 0.03488
    Epoch 2: error 130148.975, alpha = 0.03477
    Epoch 3: error 120242.958, alpha = 0.03465
    Epoch 4: error 112540.050, alpha = 0.03453
    Epoch 5: error 106864.066, alpha = 0.03441
    Epoch 6: error 102336.388, alpha = 0.03430
    Epoch 7: error 97998.764, alpha = 0.03418
    Epoch 8: error 94581.684, alpha = 0.03406
    Epoch 9: error 91675.288, alpha = 0.03395
    Epoch 10: error 89042.732, alpha = 0.03383
    Epoch 11: error 86743.940, alpha = 0.03371
    Epoch 12: error 84517.127, alpha = 0.03360
    Epoch 13: error 82460.414, alpha = 0.03348
    Epoch 14: error 80741.940, alpha = 0.03336
    Epoch 15: error 78925.016, alpha = 0.03324
    Epoch 16: error 77322.676, alpha = 0.03313
    Epoch 17: error 75869.703, alpha = 0.03301
    Epoch 18: error 74986.107, alpha = 0.03289
    Epoch 19: error 73676.221, alpha = 0.03278
    Epoch 20: error 72832.079, alpha = 0.03266
    Epoch 21: error 71578.128, alpha = 0.03254
    Epoch 22: error 70666.796, alpha = 0.03242
    Epoch 23: error 69673.525, alpha = 0.03231
    Epoch 24: error 68817.370, alpha = 0.03219
    Epoch 25: error 67810.122, alpha = 0.03207
    Epoch 26: error 66824.629, alpha = 0.03196
    Epoch 27: error 66165.420, alpha = 0.03184
    Epoch 28: error 65661.027, alpha = 0.03172
    Epoch 29: error 64943.431, alpha = 0.03161
    Epoch 30: error 64184.818, alpha = 0.03149
    Epoch 31: error 63450.960, alpha = 0.03137
    Epoch 32: error 63108.457, alpha = 0.03125
    Epoch 33: error 62177.836, alpha = 0.03114
    Epoch 34: error 61536.736, alpha = 0.03102
    Epoch 35: error 60795.149, alpha = 0.03090
    Epoch 36: error 60539.906, alpha = 0.03079
    Epoch 37: error 59747.664, alpha = 0.03067
    Epoch 38: error 59215.436, alpha = 0.03055
    Epoch 39: error 58835.278, alpha = 0.03043
    Epoch 40: error 58226.320, alpha = 0.03032
    Epoch 41: error 57989.265, alpha = 0.03020
    Epoch 42: error 57424.890, alpha = 0.03008
    Epoch 43: error 56962.105, alpha = 0.02997
    Epoch 44: error 56643.652, alpha = 0.02985
    Epoch 45: error 56341.595, alpha = 0.02973
    Epoch 46: error 55751.704, alpha = 0.02962
    Epoch 47: error 55332.606, alpha = 0.02950
    Epoch 48: error 54969.649, alpha = 0.02938
    Epoch 49: error 54552.120, alpha = 0.02926
    Epoch 50: error 54065.863, alpha = 0.02915
    Epoch 51: error 53968.124, alpha = 0.02903
    Epoch 52: error 53656.225, alpha = 0.02891
    Epoch 53: error 53125.347, alpha = 0.02880
    Epoch 54: error 52776.152, alpha = 0.02868
    Epoch 55: error 52448.319, alpha = 0.02856
    Epoch 56: error 52331.296, alpha = 0.02844
    Epoch 57: error 52331.296, alpha = 0.02833


Optimize the paragraph vectors for test set:


    alpha = 0.035
    epochs = 300
    test_errors = []
    try:
        for epoch in range(epochs):
            pvdm.alpha = max(0.0001, alpha * (1 - 1.0 * epoch / (epochs-1)))
            words, error = pvdm.train((sentence.split() for sentence in index2paragraph_test), paragraphs=paragraphs, vocab = test_paragraph_vocab, paragraphs_only=True)
            print("Epoch %d: error %.3f, alpha = %.5f" % (epoch, error, pvdm.alpha))
            if error < 1:
                break
            if epoch > 0 and error > errors[-1]:
                test_errors.append(error)
            else:
                test_errors.append(error)
    except KeyboardInterrupt:
        print("Epoch %d: error %.3f, alpha = %.5f" % (epoch, error, pvdm.alpha))

### Evaluate sentiment on test set

Get the labels in a list:


    test_labels_hash = test_trees.labels()
    test_labels = [test_labels_hash[sentence] for sentence in index2paragraph_test]

Get performance on the test set:


    print(classification_report(test_labels, reg.predict(paragraphs), labels=[0,1,2,3,4], target_names=["--", "-", "", "+", "++"]))

                 precision    recall  f1-score   support
    
             --       0.00      0.00      0.00       279
              -       0.00      0.00      0.00       633
                      0.18      1.00      0.30       389
              +       0.00      0.00      0.00       510
             ++       0.00      0.00      0.00       399
    
    avg / total       0.03      0.18      0.05      2210
    


TODO:

* Check how model does with Children's books


    
