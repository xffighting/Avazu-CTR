
import xlearn as xl
import os

featrue_version = 'v2_' 
train_path = '../Input/train'

# ftr = train_path + '/encode/'+featrue_version+'train_sample200wFFM.txt'
ftr = train_path + '/encode/'+featrue_version+'mini_trainFFM.txt'
fvl = train_path + '/encode/'+featrue_version+'validate_sample10wFFM.txt'
fte = train_path + '/encode/'+featrue_version+'test_sample10wFFM.txt'

# Training task
ffm_model = xl.create_ffm() # Use field-aware factorization machine

# On-disk training
ffm_model.setOnDisk()

ffm_model.setTrain(ftr)  # Training data
ffm_model.setValidate(fvl)  # Validation data

# param:
#  0. binary classification
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: AUC score

param =[]
param.append({'task':'binary', 'lr':0.1, 'lambda':0.0008, 'metric':'auc','epoch':20})
param.append({'task':'binary', 'lr':0.1, 'lambda':0.0009, 'metric':'auc','epoch':20})
param.append({'task':'binary', 'lr':0.1, 'lambda':0.001, 'metric':'auc','epoch':20})
param.append({'task':'binary', 'lr':0.11, 'lambda':0.001, 'metric':'auc','epoch':20})
param.append({'task':'binary', 'lr':0.12, 'lambda':0.001, 'metric':'auc','epoch':20})
param.append({'task':'binary', 'lr':0.13, 'lambda':0.001, 'metric':'auc','epoch':20})


out_path = '../Input/train/FFM_result'
if not os.path.exists(out_path):
    os.makedirs(out_path)

for i,pa in enumerate(param):
    # Start to train
    # The trained model will be stored in model.out
    ffm_model.fit(pa, out_path+'/'+featrue_version+'pa'+str(i+1)+'_model.out')

    # Prediction task
    ffm_model.setTest(fte)  # Test data
    ffm_model.setSigmoid()  # Convert output to 0-1

    # Start to predict
    # The output result will be stored in output.txt
    ffm_model.predict(out_path+'/'+featrue_version+'pa'+str(i+1)+'_model.out',
                      out_path+'/'+featrue_version+'pa'+str(i+1)+'_output.txt')
                      
