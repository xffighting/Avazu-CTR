
import xlearn as xl
import os

models = ['LR','FFM','FM']

# models = ['FFM']

featrue_version = 'v4' 
# model ='LR'

in_path = "/media/leikun/programs/AI100/AI100-Final/Input/train/" + featrue_version+ "_" + "data"
fos1 = [ in_path + '/mini_train'+featrue_version+'.ffm',
        in_path + '/validate_sample10w'+featrue_version+'.ffm',
        in_path + '/test_sample10w'+featrue_version+'.ffm',
        in_path + '/test'+featrue_version+'.ffm']

ftr = fos1[0]
fvl = fos1[1]
fte = fos1[2]

for model in models:
        
    # Training task
    if model == 'LR':
        ffm_model = xl.create_linear() # Use field-aware factorization machine

    if model == 'FFM':
        ffm_model = xl.create_ffm() 

    if model == 'FM':
        ffm_model = xl.create_fm() 

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
    # para1
    # param.append({'task':'binary', 'lr':0.686, 'lambda':0.001, 'metric':'auc','epoch':20}) #1
    # param.append({'task':'binary', 'lr':0.686, 'lambda':0.0001, 'metric':'auc','epoch':20}) # 2
    # param.append({'task':'binary', 'lr':0.1, 'lambda':0.001, 'metric':'auc','epoch':20}) #3
    # param.append({'task':'binary', 'lr':0.1, 'lambda':0.0001, 'metric':'auc','epoch':20}) #4
    # param.append({'task':'binary', 'lr':0.001, 'lambda':0.001, 'metric':'auc','epoch':20}) #5
    # param.append({'task':'binary', 'lr':0.001, 'lambda':0.0001, 'metric':'auc','epoch':20}) #6
    # param.append({'task':'binary', 'lr':0.0001, 'lambda':0.001, 'metric':'auc','epoch':20}) #7
    # param.append({'task':'binary', 'lr':0.0001, 'lambda':0.0001, 'metric':'auc','epoch':20}) #8
    # 根据para1的计算,得出lr要大,lambda要小,以便改善欠拟合

    #param.append({'task':'binary', 'lr':1, 'lambda':0.00001, 'metric':'auc','epoch':20}) #1
    #param.append({'task':'binary', 'lr':1, 'lambda':0.000001, 'metric':'auc','epoch':20}) # 2
    #param.append({'task':'binary', 'lr':1, 'lambda':0.0000001, 'metric':'auc','epoch':20}) #3
    #param.append({'task':'binary', 'lr':1, 'lambda':0.00000001, 'metric':'auc','epoch':20}) #4

    param.append({'task':'binary', 'lr':1, 'lambda':0.00001, 'metric':'auc','epoch':20}) #1
    param.append({'task':'binary', 'lr':1, 'lambda':0, 'metric':'auc','epoch':20}) # 2
    
    out_path = in_path+ '/'+model+ '_result'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for i,pa in enumerate(param):
        # Start to train
        # The trained model will be stored in model.out
        ffm_model.fit(pa, out_path+'/'+'pa'+str(i+1)+'_model.out')

        # Prediction task
        ffm_model.setTest(fte)  # Test data
        ffm_model.setSigmoid()  # Convert output to 0-1

        # Start to predict
        # The output result will be stored in output.txt
        ffm_model.predict(out_path+'/'+'pa'+str(i+1)+'_model.out',
                        out_path+'/'+'pa'+str(i+1)+'_output.txt')
                      
