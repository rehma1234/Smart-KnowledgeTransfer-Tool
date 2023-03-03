 


Smart KT – Documentation Made Simple

Group 4:
200481194 - Anjani Sonavane 
200487463 - Sanket Borkar
200494051 - Jeegniben Patel
200484628 - Rahman Armaghan
200481510 - Gurkirat Singh Saini

Objective and Scope
KT – Knowledge Transfers are integral. From giant corporations to small enterprises organize KT for various reasons. It is essential that the right information flows to the other people.
KT helps you in streamlining your knowledge which ensures that everyone in your team has the information they need to keep your business running smoothly.
Our goal is to develop a tool to get the shared text data in the video in the text format. With the use of machine learning, we can extract the text from the video. We can divide the video in frames and after doing pre-processing, text detection and text recognition, we can get the text. 
By adding the frames screenshot with text, we can document the flow also.
This will be helpful for presenter and learner both to review data any time by their convenience and use it. 
It will save time and effort. Presenter can share the document anytime to other people, and learner can easily find the specific and accurate text/data and flow without going through the whole video again.
The project's scope includes businesses from various industries. 

Scenario Formulation
It’s a very good practice to connect with each member and transfer the information with team members but at the same time it has some problems as well, at times. When someone shares the screen, it gets hard to remember accurate data. There are times when it is not feasible to make notes. Somehow, if one manages to do so then too it gets very difficult to keep account of all the facts and figures. 
In IT, majority of presenters shares their own screen during knowledge transfer because they know the data, remember the flow and all the required applications are already installed in their machine. On the other side, the learner tries to remember the data, flow, and make notes of important information. For example, if the application is banking related, then it is necessary for the learner to note down the different account numbers, portfolio names, order ids, broker name and many more with accuracy.
Even on YouTube we see lots of videos with code, but no code file present and we need to write it by our own. 
Therefore, we need a document which contains the exact text with screenshots.


Problem Statement
Knowledge transfer is one of the most important process involved in the software industry, where older employees often provide the requisite knowledge to the newer ones, so that the responsibility can be smoothly transferred to them. The problem is that the latter usually have issues regarding remembering important information passed during knowledge transfer sessions, townhall meetings and other important sessions. They usually have recordings available for them but that requires a lot of effort to go back and forth to find our desired information. They also end up missing the context even if they end up finding it correctly. 
If the noted data is not accurate, he/she needs to watch the recorded video again/needs to ask colleague/needs to ask presenter for exact data (where the presenter might not be a part of the organization at that time). This will be difficult and time consuming. 
Therefore, to overcome this real-life problem and to make the employee’s life easy, we have come up with an idea of Smart KT Documentation, which is nothing but to auto-detect the relevant text in the KT video, so that it can be used to collect and document all the information somewhere. It can be used to search and extract key data needed for our own purpose or to use it in custom reports. The generated text can then be formatted to organize the data and additionally simplified and summarized to make things more concise. If possible, we can integrate this technology with Confluence to auto-generate a new document whenever its corresponding video gets uploaded.

Machine learning Tools and Techniques
Part1: Text Extraction from KT Videos
The pipeline is as follows:
•	Text Detection: Drawing bounding boxes around the detected text
•	Character Segmentation: Segmenting the characters in the detected text
•	Character Recognition: Recognising individual characters
Methodology adopted:
EAST (An Efficient and Accurate Scene Text Detector) model for Text Detection. It is a scene text detector that directly produces word or line level predictions from full images with a single neural network. 
Pytesseract for Recognition The output bounding box of the detected text is localized and given as input to the pytesseract tool. Pytesseract is a powerful OCR tool used to recognize text.
Required libraries are OpenCV, pytesseract, imutils.
The repository contains test video to test the code.

Part2: Data Cleaning
On the received data we can perform the following steps:
Step 1: Data Pre-processing
•	Removing unnecessary punctuation, tags.
•	Removing stop words — frequent words such as “the”, “is”, etc. that do not have a specific semantic.
•	Stemming — words are reduced to a root by removing inflection by dropping unnecessary characters, usually a suffix.
•	Lemmatization — Another approach to remove inflection is by determining the part of speech and utilizing detailed database of the language.
We can use Python to do many text pre-processing operations.
•	NLTK — The Natural Language ToolKit is one of the best-known and most-used NLP libraries, useful for all sorts of tasks from tokenization, stemming, tagging, parsing, and beyond.
•	BeautifulSoup — Library for extracting data from HTML and XML documents.
Step 2: Feature Extraction
•	In text processing, words of the text represent discrete, categorical features to be used by the algorithms.
Step 3: Bag of Words (BOW): 
•	We will make the list of unique words in the text corpus called vocabulary.
•	TF-IDF value of a term as = TF * IDF (Term Frequency & Inverse Document Frequency).
Step 4: Choosing ML Algorithms:
•	Classical ML approaches like ‘Naive Bayes’ or ‘Support Vector Machines’ have been widely used.

Machine Learning Prediction & Outcomes
Text detection and recognition have been actively researched topics in computer vision for a long period of time.
An optical character recognition (OCR) algorithm built using OpenCV. In this, the user extracts text from video and displays text file. The selected region could be kept as a frame. The text in the choice region can be withdrawn utilizing OCR algorithm and made accessible to the user in a text file.
A saved image is subdivided into text line images. Those lines are separated into words and words are additionally separated into character images. The CNN is constructed using custom dataset fabricated with separation steps and accomplished using OpenCV i.e., we physically classify those segmented character images.
These images are termed as frames and can be incorporated to get the original video. So, a trouble related to video data is not that dissimilar from an image classification or an object detection trouble. There is a single additional step of extracting frames from video.
1. Import and read the video, extract frames from it, and save them as images.
2. Label a few images for training the model.
3. Identify the text using the geometry volume to derive the width and height of bounding box.
4. Make predictions for the remaining images.
We have used the research paper ‘1704.03155.pdf (arxiv.org)’ and ‘IJITCS-V11-N9-6.pdf’, in which they have conducted the test on various algorithms with different data sets, such as ICDAR 2015. It includes a total of 1500 pictures, 1000 of which are used for training and the remaining are for testing. The text regions are annotated by 4 vertices of the quadrangle, corresponding to the QUAD geometry. Another one is COCO dataset. A total of 63,686 images are annotated, in which 43,686 are chosen to be the training set and the rest 20,000 for testing. Word regions are annotated in the form of axis-aligned bounding box (AABB), which is a special case of RBOX.
The OpenCV implementation achieves a precision rate of 88.3% and a recall rate of 76.8%. This implies that the model is capable of not producing considerable false positives, but may sometimes encounter false negatives, i.e., may not detect regions containing text.
Even after training on labelled images, accuracy was not adequate. The model was not proficient to achieve well on training images itself. The model was overfitting and its outcome on unseen data was not surpassing. The limitation is that the maximal size of text instances the detector can handle is proportional to the receptive field of the network. This limits the capability of the network to predict even longer text regions like text lines running across the images.
For Part 2, according to the research paper, by using 25k for training and with 10k for test, basic NB classifier-based Sentiment Analyzer does well to give around 79% accuracy.
We have added smart text extraction from video code with one Python file in which we have performed some steps on text to clean it.






References
Mirza, A., Zeshan, O., Atif, M. et al. Detection and recognition of cursive text from video frames. J Image Video Proc. 2020, 34 (2020). https://doi.org/10.1186/s13640-020-00523-5

Goel, V., Kumar, V., Jaggi, A. S., & Nagrath, P. (2019, June). Text Extraction from Natural Scene Images using OpenCV and CNN. MECS Press. https://doi.org/10.5815/ijitcs.2019.09.06

Jesmeen M. Z. H., Hossen, J., Sayeed, S., Ho, C. K., Tawsif, K., Rahnam, A., & Arif, E. M. H. (2018, June). A Survey on Cleaning Dirty Data Using Machine Learning Paradigm for Big Data Analytics. Indonesian Journal of Electrical Engineering and Computer Science. https://doi.org/10.11591/ijeecs.v10.i3.pp1234-1243

Ma, D., Lin, Q., & Zhang, T. (2011). Mobile Camera Based Text Detection and Translation. Department of Mechanical Engineering Stanford University.

Pooja, & Dhir, R. (2016, May). Video Text Extraction and Recognition: A Survey. IEEE. https://doi.org/10.1109/WiSPNET.2016.7566360

Zhou, X., Yao, C., Wen, H., Wang, Y., Zhou, S., He, W., & Liang, J. (2017, July). EAST: An Efficient and Accurate Scene Text Detector. Megvii Technology Inc., Beijing, China.
