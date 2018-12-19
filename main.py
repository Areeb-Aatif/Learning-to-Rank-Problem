#!/usr/bin/env python
# coding: utf-8

# In[5]:

from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt


# In[6]:


maxAcc = 0.0
maxIter = 0
C_Lambda = 0.05
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 40                      # Number of clusters for KMeans Algorithm or the number of basis functions.
PHI = []                    # Vector of M basis functions.
IsSynthetic = False


# In[158]:

# This function takes the Querylevelnorm_t.csv as input and returns the target vector t.
def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
    print("Raw Training Generated..")
    return t

# This function takes the Querylevelnorm_X.csv as input and returns a matrix (dataMatrix),
# consisting of real valued vectors.
# Here isSynthetic = False is used to delete coloumns from 5 to 9 which consists of all zeros.
def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   
    
    if IsSynthetic == False :
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1)
    dataMatrix = np.transpose(dataMatrix)     
    print ("Data Matrix Generated..")
    return dataMatrix

# This function divides the target vector and returns 80% of target values as a vector t
# Here vector t is used as training target ( i.e. to train the model )   
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    print(str(TrainingPercent) + "% Training Target Generated..")
    return t

# This function divides the input data matrix and returns 80% of input data matrix as a matrix d2
# Here matrix d2 is used as a training data ( i.e. to train the model )
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

# This function takes rawData as input data file and ValPercent as 10% and returns dataMatrix used as
# validation data set.   
# ValPercent decides how much of the total validation data must be generated from the total set.
# This function is again used for creating test data set which is 10% of the total set.
def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

# This function takes rawData as input data file and ValPercent as 10% and returns vector t used as
# validation target set.   
# ValPercent decides how much of the total validation data must be generated from the total set.
# This function is again used for creating test target vector which is 10% of the total set.
def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

# This function computes BigSigma matrix of dimensions 41x41.
# It computes variance for each of the 41 features and stores it in BigSigma matrix.
def GenerateBigSigma(Data, MuMatrix, TrainingPercent, IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))
    
    # Storing variance at only the diagonal terms. Constraining Sigma to be a diagonal matrix.
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    #print ("BigSigma Generated..")
    return BigSigma

# This function converts input vector x to a specific scalar value 
# based on the formula in project description. 
def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

# This function multiplies the scalar value obtained from GetScalar function with -0.5
# and then takes the exponential value to generate PHI for a particular input index.    
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

# Function to find the PHI matrix.
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))   
    # Initializing the length of PHI matrix.      
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    # Calculating Inverse of BigSigma.
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

# This function calculates weight vector w using the formula for closed form solution 
# with least squared regularization.
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W

# This function outputs Y for a particular value of x,w (Linear Regression). 
def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

# This function calculates Erms for a particular set of data. 
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# ## Fetch and Prepare Dataset

# In[127]:

# Querylevelnorm_t.csv stores all the raw target values t extracted from the Querylevelnorm.txt.
# GetTargetVector extracts the target values t in a vector form (RawTarget) from the csv file.
RawTarget = GetTargetVector('Querylevelnorm_t.csv')
# Querylevelnorm_X.csv stores all input values x (Vector consisting of query-document pair), 
# extracted from the Querylevelnorm.txt.
# GetTargetData extracts the input vectors x in a matrix form (RawData) from the csv file.
RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic)


# ## Prepare Training Data

# In[128]:

# TraningTarget consists of 80% of target values used to train our model.
TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
# TraningData consists of 80% of the total input data vectors used to train our model.
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)

# AnyVector.shape prints the dimensions of a vector. If it is a 1D array, it prints (n,) where 
# n is number of total elements. If it is 2D array it prints (n, m) where n is the number of rows
# and m is the number of columns.
# Here TrainingTarget.shape prints (55699,). Here 55699 is the 80% of total target values.
print(TrainingTarget.shape)
# Here TrainingData.shape prints (41, 55699).
print(TrainingData.shape)


# ## Prepare Validation Data

# In[129]:

# Here we are generating the validation target vector from the total target vector set.
ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
# Here we are generating the validation data set from the total input data set.
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
# Here ValDataAct.shape prints (6962,).
print(ValDataAct.shape)
# Here ValData.shape prints (41, 6962).
print(ValData.shape)


# ## Prepare Test Data

# In[130]:

# Here we are passing test percentage to the same validation target data function to create test target.
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
# Here we are passing test percentage to the same validation data function to create test data set.
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
# TestDataAct.shape prints (6961,).
print(TestDataAct.shape)
# TestData.shape prints (41, 6961).
print(TestData.shape)


# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

# In[155]:

ErmsArr = []
AccuracyArr = []

# Taking M=10 as number of clusters, we apply kmeans clustering algorithm on the training data.
# Here KMeans is a function present in sklearn.cluster library, which divides the data into 
# M clusters with each cluster having a centroid. 
# Points in each cluster are said to be more similar. 
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
# Mu stores centers of M basis functions.
Mu = kmeans.cluster_centers_

# Calculating BigSigma (Gaussian radial basis function) which is a part of the formula of PHI.
BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent, IsSynthetic)
# Calculating PHI for training data set using the formula given in the project description.
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
# Calculating weight vector w in closed form solution. Here we are finding closed form solution using 
# least squared regularization. Hence we pass Lamda to get w. 
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
# Similar to Training PHI we compute TEST_PHI for the test data.
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
# Similar to Training PHI we compute VAL_PHI for the validation data.
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


# In[156]:

print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)


# ## Finding Erms on training, validation and test set 
# Here we are calculating Root mean square error.

# In[159]:

# We calculate y from the basic formula of linear regression y(x,w) = transpose(PHI)*w.  
TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

# Here we calculate Erms for each training, validation and test set.
TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


# In[160]:


print ('UBITname      = *****')
print ('Person Number = *****')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = 40 \nLambda = 0.05")
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))


# ## Gradient Descent solution for Linear Regression
# SGD is an iterative method for optimizing the cost function. 
# It takes only one training sample at a time.

# In[138]:


print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')


# In[ ]:

W_Now        = np.dot(220, W)       # Random initial value w(0) 
La           = 2                    # Lamda 
learningRate = 0.007                # It decides how big each update step would be
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

# Calculating Training, Validation and Testing Erms for the first 400 input vectors of training data. 
for i in range(0,400):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    # Here we compute each term present in the formula of Stochastic Gradient Descent 
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W                 # Updates value of w at each iteration.
    W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
    # We calculate Erms for the training data set using w calculated in this iteration.
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
    # We calculate Erms for the Validation data set using w calculated in this iteration.
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    # We calculate Erms for the test data set using w calculated in this iteration.
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


# In[ ]:

# Erms Training = Minimum of the list (L_Erms_TR). L_Erms_TR stores training accuracy at each iteration.
# Erms Validation = Minimum of the list (Erms Validation).L_Erms_Val stores validation accuracy at each iteration.
# Erms Test = Minimum of the list (Erms Test). L_Erms_Test stores test accuracy at each iteration.
print ('----------Gradient Descent Solution--------------------')
print ("M = 40 \nLambda  = 2\neta=0.007")
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
