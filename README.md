# Deep Learning Papers Reading Roadmap

>If you are a newcomer to the Deep Learning area, the first question you may have is "Which paper should I start reading from?"

>Here is a reading roadmap of Deep Learning papers!

The roadmap is constructed in accordance with the following four guidelines:

- From outline to detail
- From old to state-of-the-art
- from generic to specific areas
- focus on state-of-the-art

You will find many papers that are quite new but really worth reading.

I would continue adding papers to this roadmap.


---------------------------------------

# Deep Learning History and Basics

## Book

- [ ] **Deep learning**. Bengio, Yoshua, Ian J. Goodfellow, and Aaron Courville. [[html]](http://www.deeplearningbook.org/) :star::star::star::star::star:
    - Deep Learning Bible, you can read this book while reading following papers.

## Survey

- [ ] **Deep learning**. LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. [[pdf]](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) :star::star::star::star::star:
    - Three Giants' Survey

## Deep Belief Network(DBN)(Milestone of Deep Learning Eve)

- [ ] **A fast learning algorithm for deep belief nets**. Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. [[pdf]](http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf) :star::star::star:
    - Deep Learning Eve

- [ ] **Reducing the dimensionality of data with neural networks**. Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. [[pdf]](http://www.cs.toronto.edu/~hinton/science.pdf) :star::star::star:
    - Milestone, Show the promise of deep learning

## ImageNet Evolution（Deep Learning broke out from here）

- [ ] **Imagenet classification with deep convolutional neural networks**. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) :star::star::star::star::star:
    - AlexNet, Deep Learning Breakthrough

- [ ] **Very deep convolutional networks for large-scale image recognition**. Simonyan, Karen, and Andrew Zisserman. [[pdf]](https://arxiv.org/pdf/1409.1556.pdf) :star::star::star:
    - VGGNet,Neural Networks become very deep!

- [ ] **Going deeper with convolutions**. Szegedy, Christian, et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) :star::star::star:
    - GoogLeNet

- [ ] **Deep residual learning for image recognition**. He, Kaiming, et al. [[pdf]](https://arxiv.org/pdf/1512.03385.pdf) :star::star::star::star::star:
    - ResNet,Very very deep networks, CVPR best paper

## Speech Recognition Evolution

- [ ] **Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups**. Hinton, Geoffrey, et al. [[pdf]](http://cs224d.stanford.edu/papers/maas_paper.pdf):star::star::star::star:
    - Breakthrough in speech recognition

- [ ] **Speech recognition with deep recurrent neural networks**. Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. [[pdf]](http://arxiv.org/pdf/1303.5778.pdf):star::star::star:
    - RNN

- [ ] **Towards End-To-End Speech Recognition with Recurrent Neural Networks**. Graves, Alex, and Navdeep Jaitly. [[pdf]](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf):star::star::star:

- [ ] **Fast and accurate recurrent neural network acoustic models for speech recognition**. Sak, Haşim, et al. [[pdf]](http://arxiv.org/pdf/1507.06947) :star::star::star:
    - Google Speech Recognition System

- [ ] **Deep speech 2: End-to-end speech recognition in english and mandarin**. Amodei, Dario, et al. [[pdf]](https://arxiv.org/pdf/1512.02595.pdf) :star::star::star::star:
    - Baidu Speech Recognition System

- [ ] **Achieving Human Parity in Conversational Speech Recognition**. W. Xiong, J. Droppo, X. Huang, F. Seide, M. Seltzer, A. Stolcke, D. Yu, G. Zweig [[pdf]](https://arxiv.org/pdf/1610.05256v1) :star::star::star::star:
    - State-of-the-art in speech recognition, Microsoft

>After reading above papers, you will have a basic understanding of the Deep Learning history, the basic architectures of Deep Learning model(including CNN, RNN, LSTM) and how deep learning can be applied to image and speech recognition issues. The following papers will take you in-depth understanding of the Deep Learning method, Deep Learning in different areas of application and the frontiers. I suggest that you can choose the following papers based on your interests and research direction.

# Deep Learning Method

## Model

- [ ] **Improving neural networks by preventing co-adaptation of feature detectors**. Hinton, Geoffrey E., et al. [[pdf]](https://arxiv.org/pdf/1207.0580.pdf) :star::star::star:
    - Dropout

- [ ] **Dropout: a simple way to prevent neural networks from overfitting**. Srivastava, Nitish, et al. [[pdf]](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) :star::star::star:

- [ ] **Batch normalization: Accelerating deep network training by reducing internal covariate shift**. Ioffe, Sergey, and Christian Szegedy. [[pdf]](http://arxiv.org/pdf/1502.03167) :star::star::star::star:
    - An outstanding Work in 2015

- [ ] **Layer normalization**. Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. [[pdf]](https://arxiv.org/pdf/1607.06450.pdf?utm_source=sciontist.com&utm_medium=refer&utm_campaign=promote) :star::star::star::star:
    - Update of Batch Normalization

- [ ] **Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to+ 1 or−1**. Courbariaux, Matthieu, et al. [[pdf]](https://pdfs.semanticscholar.org/f832/b16cb367802609d91d400085eb87d630212a.pdf)  :star::star::star:
    - New Model,Fast

- [ ] **Decoupled neural interfaces using synthetic gradients**. Jaderberg, Max, et al. [[pdf]](https://arxiv.org/pdf/1608.05343) :star::star::star::star::star:
    - Innovation of Training Method,Amazing Work

- [ ] Chen, Tianqi, Ian Goodfellow, and Jonathon Shlens. "Net2net: Accelerating learning via knowledge transfer." arXiv preprint arXiv:1511.05641 (2015). [[pdf]](https://arxiv.org/abs/1511.05641) :star::star::star:
    - Modify previously trained network to reduce training epochs

- [ ] Wei, Tao, et al. "Network Morphism." arXiv preprint arXiv:1603.01670 (2016). [[pdf]](https://arxiv.org/abs/1603.01670) :star::star::star:
    - Modify previously trained network to reduce training epochs

## Optimization

- [ ] **On the importance of initialization and momentum in deep learning**. Sutskever, Ilya, et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v28/sutskever13.pdf) :star::star:
    - Momentum optimizer

- [ ] **Adam: A method for stochastic optimization**. Kingma, Diederik, and Jimmy Ba. [[pdf]](http://arxiv.org/pdf/1412.6980) :star::star::star:
    - Maybe used most often currently

- [ ] **Learning to learn by gradient descent by gradient descent**. Andrychowicz, Marcin, et al. [[pdf]](https://arxiv.org/pdf/1606.04474) :star::star::star::star::star:
    - Neural Optimizer,Amazing Work

- [ ] **Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding**. Han, Song, Huizi Mao, and William J. Dally. [[pdf]](https://pdfs.semanticscholar.org/5b6c/9dda1d88095fa4aac1507348e498a1f2e863.pdf) :star::star::star::star::star:
    - ICLR best paper, new direction to make NN running fast,DeePhi Tech Startup

- [ ] **SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size**. Iandola, Forrest N., et al. [[pdf]](http://arxiv.org/pdf/1602.07360) :star::star::star::star:
    - Also a new direction to optimize NN,DeePhi Tech Startup

## Unsupervised Learning / Deep Generative Model

- [ ] **Building high-level features using large scale unsupervised learning**. Le, Quoc V. [[pdf]](http://arxiv.org/pdf/1112.6209.pdf&embed) :star::star::star::star:
    - Milestone, Andrew Ng, Google Brain Project, Cat


- [ ] **Auto-encoding variational bayes**. Kingma, Diederik P., and Max Welling. [[pdf]](http://arxiv.org/pdf/1312.6114) :star::star::star::star:
    - VAE

- [ ] **Generative adversarial nets**. Goodfellow, Ian, et al. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) :star::star::star::star::star:
    - GAN,super cool idea

- [ ] **Unsupervised representation learning with deep convolutional generative adversarial networks**. Radford, Alec, Luke Metz, and Soumith Chintala. [[pdf]](http://arxiv.org/pdf/1511.06434) :star::star::star::star:
    - DCGAN

- [ ] **DRAW: A recurrent neural network for image generation**. Gregor, Karol, et al. [[pdf]](http://jmlr.org/proceedings/papers/v37/gregor15.pdf) :star::star::star::star::star:
    - VAE with attention, outstanding work

- [ ] **Pixel recurrent neural networks**. Oord, Aaron van den, Nal Kalchbrenner, and Koray Kavukcuoglu. [[pdf]](http://arxiv.org/pdf/1601.06759) :star::star::star::star:
    - PixelRNN

- [ ] Oord, Aaron van den, et al. "Conditional image generation with PixelCNN decoders." arXiv preprint arXiv:1606.05328 (2016). [[pdf]](https://arxiv.org/pdf/1606.05328) :star::star::star::star:
    - PixelCNN

## RNN / Sequence-to-Sequence Model

- [ ] **Generating sequences with recurrent neural networks**. Graves, Alex. [[pdf]](http://arxiv.org/pdf/1308.0850) :star::star::star::star:
    - LSTM, very nice generating result, show the power of RNN

- [ ] **Learning phrase representations using RNN encoder-decoder for statistical machine translation**. Cho, Kyunghyun, et al. [[pdf]](http://arxiv.org/pdf/1406.1078) :star::star::star::star:
    - First Seq-to-Seq Paper

- [ ] **Sequence to sequence learning with neural networks**. Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. [[pdf]](https://arxiv.org/pdf/1409.3215.pdf) :star::star::star::star::star:
    - Outstanding Work

- [ ] **Neural Machine Translation by Jointly Learning to Align and Translate**. Bahdanau, Dzmitry, KyungHyun Cho, and Yoshua Bengio. [[pdf]](https://arxiv.org/pdf/1409.0473v7.pdf) :star::star::star::star:

- [ ] **A neural conversational model**. Vinyals, Oriol, and Quoc Le. [[pdf]](http://arxiv.org/pdf/1506.05869.pdf%20(http://arxiv.org/pdf/1506.05869.pdf)) :star::star::star:
    - Seq-to-Seq on Chatbot

## Neural Turing Machine

- [ ] **Neural turing machines**. Graves, Alex, Greg Wayne, and Ivo Danihelka. [[pdf]](http://arxiv.org/pdf/1410.5401.pdf) :star::star::star::star::star:
    - Basic Prototype of Future Computer

- [ ] **Reinforcement learning neural Turing machines**. Zaremba, Wojciech, and Ilya Sutskever. [[pdf]](https://pdfs.semanticscholar.org/f10e/071292d593fef939e6ef4a59baf0bb3a6c2b.pdf) :star::star::star:

- [ ] **Memory networks**. Weston, Jason, Sumit Chopra, and Antoine Bordes. [[pdf]](http://arxiv.org/pdf/1410.3916) :star::star::star:


- [ ] **End-to-end memory networks**. Sukhbaatar, Sainbayar, Jason Weston, and Rob Fergus. [[pdf]](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf) :star::star::star::star:

- [ ] **Pointer networks**. Vinyals, Oriol, Meire Fortunato, and Navdeep Jaitly. [[pdf]](http://papers.nips.cc/paper/5866-pointer-networks.pdf) :star::star::star::star:

- [ ] **Hybrid computing using a neural network with dynamic external memory**. Graves, Alex, et al. [[pdf]](https://www.dropbox.com/s/0a40xi702grx3dq/2016-graves.pdf) :star::star::star::star::star:
    - Milestone,combine above papers' ideas

## Deep Reinforcement Learning

- [ ] **Playing atari with deep reinforcement learning**. Mnih, Volodymyr, et al. [[pdf]](http://arxiv.org/pdf/1312.5602.pdf)) :star::star::star::star:
    - First Paper named deep reinforcement learning

- [ ] **Human-level control through deep reinforcement learning**. Mnih, Volodymyr, et al. [[pdf]](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) :star::star::star::star::star:
    - Milestone

- [ ] **Dueling network architectures for deep reinforcement learning**. Wang, Ziyu, Nando de Freitas, and Marc Lanctot. [[pdf]](http://arxiv.org/pdf/1511.06581)  :star::star::star::star:
    - ICLR best paper,great idea

- [ ] **Asynchronous methods for deep reinforcement learning**. Mnih, Volodymyr, et al. [[pdf]](http://arxiv.org/pdf/1602.01783) :star::star::star::star::star:
    - State-of-the-art method

- [ ] **Continuous control with deep reinforcement learning**. Lillicrap, Timothy P., et al. [[pdf]](http://arxiv.org/pdf/1509.02971) :star::star::star::star:
    - DDPG

- [ ] **Continuous Deep Q-Learning with Model-based Acceleration**. Gu, Shixiang, et al. [[pdf]](http://arxiv.org/pdf/1603.00748) :star::star::star::star:
    - NAF

- [ ] **Trust region policy optimization**. Schulman, John, et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v37/schulman15.pdf) :star::star::star::star:
    - TRPO

- [ ] **Mastering the game of Go with deep neural networks and tree search**. Silver, David, et al. [[pdf]](http://willamette.edu/~levenick/cs448/goNature.pdf) :star::star::star::star::star:
    - AlphaGo

## Deep Transfer Learning / Lifelong Learning / especially for RL

- [ ] **Deep Learning of Representations for Unsupervised and Transfer Learning**. Bengio, Yoshua. [[pdf]](http://www.jmlr.org/proceedings/papers/v27/bengio12a/bengio12a.pdf) :star::star::star:
    - A Tutorial

- [ ] **Lifelong Machine Learning Systems: Beyond Learning Algorithms**. Silver, Daniel L., Qiang Yang, and Lianghao Li. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.696.7800&rep=rep1&type=pdf)  :star::star::star:
    - A brief discussion about lifelong learning

- [ ] **Distilling the knowledge in a neural network**. Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. [[pdf]](http://arxiv.org/pdf/1503.02531) :star::star::star::star:
    - Godfather's Work

- [ ] **Policy distillation**. Rusu, Andrei A., et al. [[pdf]](http://arxiv.org/pdf/1511.06295) :star::star::star:
    - RL domain

- [ ] **Actor-mimic: Deep multitask and transfer reinforcement learning**. Parisotto, Emilio, Jimmy Lei Ba, and Ruslan Salakhutdinov. [[pdf]](http://arxiv.org/pdf/1511.06342) :star::star::star:
    - RL domain

- [ ] **Progressive neural networks**. Rusu, Andrei A., et al. [[pdf]](https://arxiv.org/pdf/1606.04671) :star::star::star::star::star:
    - Outstanding Work, A novel idea


## One Shot Deep Learning

- [ ] **Human-level concept learning through probabilistic program induction**. Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. [[pdf]](http://clm.utexas.edu/compjclub/wp-content/uploads/2016/02/lake2015.pdf) :star::star::star::star::star:
    - No Deep Learning,but worth reading

- [ ] **Siamese Neural Networks for One-shot Image Recognition**. Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. [[pdf]](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf) :star::star::star:

- [ ] **One-shot Learning with Memory-Augmented Neural Networks**. Santoro, Adam, et al. [[pdf]](http://arxiv.org/pdf/1605.06065) :star::star::star::star:
    - A basic step to one shot learning

- [ ] **Matching Networks for One Shot Learning**. Vinyals, Oriol, et al. [[pdf]](https://arxiv.org/pdf/1606.04080) :star::star::star:

- [ ] **Low-shot visual object recognition**. Hariharan, Bharath, and Ross Girshick. [[pdf]](http://arxiv.org/pdf/1606.02819) :star::star::star::star:
    - A step to large data


# Applications

## Natural Language Processing (NLP)

- [ ] **Joint Learning of Words and Meaning Representations for Open-Text Semantic Parsing**. Antoine Bordes, et al. [[pdf]](https://www.hds.utc.fr/~bordesan/dokuwiki/lib/exe/fetch.php?id=en%3Apubli&cache=cache&media=en:bordes12aistats.pdf) :star::star::star::star:
- [ ] **Distributed representations of words and phrases and their compositionality**. Mikolov, et al. [[pdf]](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) :star::star::star:
    - word2vec
- [ ] **“Sequence to sequence learning with neural networks**. Sutskever, et al. [[pdf]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) :star::star::star:
- [ ] **“Ask Me Anything: Dynamic Memory Networks for Natural Language Processing**. Ankit Kumar, et al. [[pdf]](https://arxiv.org/abs/1506.07285) :star::star::star::star:
- [ ] **Character-Aware Neural Language Models**. Yoon Kim, et al. [[pdf]](https://arxiv.org/abs/1508.06615) :star::star::star::star:
- [ ] **Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks**. Jason Weston, et al. [[pdf]](https://arxiv.org/abs/1502.05698) :star::star::star:
    - bAbI tasks
- [ ] **Teaching Machines to Read and Comprehend**. Karl Moritz Hermann, et al. [[pdf]](https://arxiv.org/abs/1506.03340) :star::star:
    - CNN/DailyMail cloze style questions
- [ ] **Very Deep Convolutional Networks for Natural Language Processing**. Alexis Conneau, et al. [[pdf]](https://arxiv.org/abs/1606.01781) :star::star::star:
    - state-of-the-art in text classification
- [ ] **Bag of Tricks for Efficient Text Classification**. Armand Joulin, et al. [[pdf]](https://arxiv.org/abs/1607.01759) :star::star::star:
    - slightly worse than state-of-the-art, but a lot faster

## Object Detection

- [ ] **Deep neural networks for object detection**. Szegedy, Christian, Alexander Toshev, and Dumitru Erhan. [[pdf]](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf) :star::star::star:

- [ ] **Rich feature hierarchies for accurate object detection and semantic segmentation**. Girshick, Ross, et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) :star::star::star::star::star:
    - RCNN

- [ ] **Spatial pyramid pooling in deep convolutional networks for visual recognition**. He, Kaiming, et al. [[pdf]](http://arxiv.org/pdf/1406.4729) :star::star::star::star:
    - SPPNet

- [ ] **Fast r-cnn**. Girshick, Ross. [[pdf]](https://pdfs.semanticscholar.org/8f67/64a59f0d17081f2a2a9d06f4ed1cdea1a0ad.pdf) :star::star::star::star:

- [ ] **Faster R-CNN: Towards real-time object detection with region proposal networks**. Ren, Shaoqing, et al. [[pdf]](https://arxiv.org/pdf/1506.01497.pdf) :star::star::star::star:

- [ ] **You only look once: Unified, real-time object detection**. Redmon, Joseph, et al. [[pdf]](http://homes.cs.washington.edu/~ali/papers/YOLO.pdf) :star::star::star::star::star:
    - YOLO,Oustanding Work, really practical

- [ ] **SSD: Single Shot MultiBox Detector**. Liu, Wei, et al. [[pdf]](http://arxiv.org/pdf/1512.02325) :star::star::star:

- [ ] Dai, Jifeng, et al. "**R-FCN: Object Detection via
Region-based Fully Convolutional Networks**" [[pdf]](https://arxiv.org/abs/1605.06409) :star::star::star::star:

- [ ] **Mask R-CNN**. He, Gkioxari, et al. [[pdf]](https://arxiv.org/abs/1703.06870) :star::star::star::star:
## Visual Tracking

- [ ] **Learning a deep compact image representation for visual tracking**. Wang, Naiyan, and Dit-Yan Yeung. [[pdf]](http://papers.nips.cc/paper/5192-learning-a-deep-compact-image-representation-for-visual-tracking.pdf) :star::star::star:
    - First Paper to do visual tracking using Deep Learning,DLT Tracker

- [ ] **Transferring rich feature hierarchies for robust visual tracking**. Wang, Naiyan, et al. [[pdf]](http://arxiv.org/pdf/1501.04587) :star::star::star::star:
    - SO-DLT

- [ ] **Visual tracking with fully convolutional networks**. Wang, Lijun, et al. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Wang_Visual_Tracking_With_ICCV_2015_paper.pdf) :star::star::star::star:
    - FCNT

- [ ] **Learning to Track at 100 FPS with Deep Regression Networks**. Held, David, Sebastian Thrun, and Silvio Savarese. [[pdf]](http://arxiv.org/pdf/1604.01802) :star::star::star::star:
    - GOTURN,Really fast as a deep learning method,but still far behind un-deep-learning methods

- [ ] **Fully-Convolutional Siamese Networks for Object Tracking**. Bertinetto, Luca, et al. [[pdf]](https://arxiv.org/pdf/1606.09549) :star::star::star::star:
    - SiameseFC,New state-of-the-art for real-time object tracking

- [ ] **Beyond Correlation Filters: Learning Continuous Convolution Operators for Visual Tracking**. Martin Danelljan, Andreas Robinson, Fahad Khan, Michael Felsberg. [[pdf]](http://www.cvl.isy.liu.se/research/objrec/visualtracking/conttrack/C-COT_ECCV16.pdf) :star::star::star::star:
    - C-COT

- [ ] **Modeling and Propagating CNNs in a Tree Structure for Visual Tracking**. Nam, Hyeonseob, Mooyeol Baek, and Bohyung Han. [[pdf]](https://arxiv.org/pdf/1608.07242) :star::star::star::star:
    - VOT2016 Winner,TCNN

## Image Caption
- [ ] **Every picture tells a story: Generating sentences from images**. Farhadi,Ali,etal. [[pdf]](https://www.cs.cmu.edu/~afarhadi/papers/sentence.pdf) :star::star::star:

- [ ] **Baby talk: Understanding and generating image descriptions**. Kulkarni, Girish, et al. [[pdf]](http://tamaraberg.com/papers/generation_cvpr11.pdf):star::star::star::star:

- [ ] **Show and tell: A neural image caption generator**. Vinyals, Oriol, et al. [[pdf]](https://arxiv.org/pdf/1411.4555.pdf):star::star::star:

- [ ] **Long-term recurrent convolutional networks for visual recognition and description**. Donahue, Jeff, et al. [[pdf]](https://arxiv.org/pdf/1411.4389.pdf)

- [ ] **Deep visual-semantic alignments for generating image descriptions**. Karpathy, Andrej, and Li Fei-Fei. [[pdf]](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf):star::star::star::star::star:

- [ ] **Deep fragment embeddings for bidirectional image sentence mapping**. Karpathy, Andrej, Armand Joulin, and Fei Fei F. Li. [[pdf]](https://arxiv.org/pdf/1406.5679v1.pdf):star::star::star::star:

- [ ] **From captions to visual concepts and back**. Fang, Hao, et al. [[pdf]](https://arxiv.org/pdf/1411.4952v3.pdf):star::star::star::star::star:

- [ ] **Learning a recurrent visual representation for image caption generation**. Chen, Xinlei, and C. Lawrence Zitnick. [[pdf]](https://arxiv.org/pdf/1411.5654v1.pdf):star::star::star::star:

- [ ] **Deep captioning with multimodal recurrent neural networks (m-rnn)**. Mao, Junhua, et al. [[pdf]](https://arxiv.org/pdf/1412.6632v5.pdf):star::star::star:

- [ ] **Show, attend and tell: Neural image caption generation with visual attention**. Xu, Kelvin, et al. [[pdf]](https://arxiv.org/pdf/1502.03044v3.pdf):star::star::star::star::star:

## Machine Translation

> Some milestone papers are listed in RNN / Seq-to-Seq topic.

- [ ] **Addressing the rare word problem in neural machine translation**. Luong, Minh-Thang, et al. [[pdf]](http://arxiv.org/pdf/1410.8206) :star::star::star::star:


- [ ] **Neural Machine Translation of Rare Words with Subword Units**. Sennrich, et al. [[pdf]](https://arxiv.org/pdf/1508.07909.pdf):star::star::star:

- [ ] **Effective approaches to attention-based neural machine translation**. Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. [[pdf]](http://arxiv.org/pdf/1508.04025) :star::star::star::star:

- [ ] **A Character-Level Decoder without Explicit Segmentation for Neural Machine Translation**. Chung, et al. [[pdf]](https://arxiv.org/pdf/1603.06147.pdf):star::star:

- [ ] **Fully Character-Level Neural Machine Translation without Explicit Segmentation**. Lee, et al. [[pdf]](https://arxiv.org/pdf/1610.03017.pdf):star::star::star::star::star:

- [ ] **Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation**. Wu, Schuster, Chen, Le, et al. [[pdf]](https://arxiv.org/pdf/1609.08144v2.pdf) :star::star::star::star:
    - Milestone

## Robotics

- [ ] **Evolving large-scale neural networks for vision-based reinforcement learning**. Koutník, Jan, et al. [[pdf]](http://repository.supsi.ch/4550/1/koutnik2013gecco.pdf) :star::star::star:

- [ ] **End-to-end training of deep visuomotor policies**. Levine, Sergey, et al. [[pdf]](http://www.jmlr.org/papers/volume17/15-522/15-522.pdf) :star::star::star::star::star:

- [ ] **Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours**. Pinto, Lerrel, and Abhinav Gupta. [[pdf]](http://arxiv.org/pdf/1509.06825) :star::star::star:

- [ ] **Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection**. Levine, Sergey, et al. [[pdf]](http://arxiv.org/pdf/1603.02199) :star::star::star::star:

- [ ] **Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning**. Zhu, Yuke, et al. [[pdf]](https://arxiv.org/pdf/1609.05143) :star::star::star::star:

- [ ] **Collective Robot Reinforcement Learning with Distributed Asynchronous Guided Policy Search**. Yahya, Ali, et al. [[pdf]](https://arxiv.org/pdf/1610.00673) :star::star::star::star:

- [ ] **Deep Reinforcement Learning for Robotic Manipulation**. Gu, Shixiang, et al. [[pdf]](https://arxiv.org/pdf/1610.00633) :star::star::star::star:

- [ ] **Sim-to-Real Robot Learning from Pixels with Progressive Nets**. A Rusu, M Vecerik, Thomas Rothörl, N Heess, R Pascanu, R Hadsell [[pdf]](https://arxiv.org/pdf/1610.04286.pdf) :star::star::star::star:

- [ ] **Learning to navigate in complex environments**. Mirowski, Piotr, et al. [[pdf]](https://arxiv.org/pdf/1611.03673) :star::star::star::star:

## Art

- [ ] **Inceptionism: Going Deeper into Neural Networks**. Mordvintsev, Alexander; Olah, Christopher; Tyka, Mike (2015). [[html]](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
    - Deep Dream
:star::star::star::star:

- [ ] **A neural algorithm of artistic style**. Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. [[pdf]](http://arxiv.org/pdf/1508.06576) :star::star::star::star::star:
    - Outstanding Work, most successful method currently

- [ ] **Generative Visual Manipulation on the Natural Image Manifold**. Zhu, Jun-Yan, et al. [[pdf]](https://arxiv.org/pdf/1609.03552) :star::star::star::star:
    - iGAN

- [ ] **Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks**. Champandard, Alex J. [[pdf]](http://arxiv.org/pdf/1603.01768) :star::star::star::star:
    - Neural Doodle

- [ ] **Colorful Image Colorization**. Zhang, Richard, Phillip Isola, and Alexei A. Efros. [[pdf]](http://arxiv.org/pdf/1603.08511) :star::star::star::star:

- [ ] **Perceptual losses for real-time style transfer and super-resolution**. Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. [[pdf]](https://arxiv.org/pdf/1603.08155.pdf) :star::star::star::star:

- [ ] **A learned representation for artistic style**. Vincent Dumoulin, Jonathon Shlens and Manjunath Kudlur. [[pdf]](https://arxiv.org/pdf/1610.07629v1.pdf) :star::star::star::star:

- [ ] **Controlling Perceptual Factors in Neural Style Transfer**. Gatys, Leon and Ecker, et al [[pdf]](https://arxiv.org/pdf/1611.07865.pdf):star::star::star::star:
    - control style transfer over spatial location,colour information and across spatial scale

- [ ] **Texture Networks: Feed-forward Synthesis of Textures and Stylized Images**. Ulyanov, Dmitry and Lebedev, Vadim, et al. [[pdf]](http://arxiv.org/abs/1603.03417) :star::star::star::star:
    - texture generation and style transfer


## Object Segmentation

- [ ] J. Long, E. Shelhamer, and T. Darrell, “**Fully convolutional networks for semantic segmentation**" [[pdf]](https://arxiv.org/pdf/1411.4038v2.pdf) :star::star::star::star::star:

- [ ] **Semantic image segmentation with deep convolutional nets and fully connected crfs**. L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. [[pdf]](https://arxiv.org/pdf/1606.00915v1.pdf) :star::star::star::star::star:

- [ ] **Learning to segment object candidates.**. Pinheiro, P.O., Collobert, R., Dollar, P. [[pdf]](https://arxiv.org/pdf/1506.06204v2.pdf) :star::star::star::star:

- [ ] **Instance-aware semantic segmentation via multi-task network cascades**. Dai, J., He, K., Sun, J. [[pdf]](https://arxiv.org/pdf/1512.04412v1.pdf) :star::star::star:

- [ ] **Instance-sensitive Fully Convolutional Networks**. Dai, J., He, K., Sun, J. [[pdf]](https://arxiv.org/pdf/1603.08678v1.pdf) :star::star::star:


