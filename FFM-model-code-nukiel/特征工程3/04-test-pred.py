
import xlearn as xl
import os
import pandas as pd

featrue_version = ['v3_']

submission = 'y'
subparam = 3
train_path = '../Input/train'

param =[]
param.append({'task':'binary', 'lr':0.1, 'lambda':0.002, 'metric':'auc','epoch':20})
param.append({'task':'binary', 'lr':0.01, 'lambda':0.002, 'metric':'auc','epoch':20})
param.append({'task':'binary', 'lr':0.001, 'lambda':0.002, 'metric':'auc','epoch':20})
param.append({'task':'binary', 'lr':0.1, 'lambda':0.001, 'metric':'auc','epoch':20})
param.append({'task':'binary', 'lr':0.01, 'lambda':0.003, 'metric':'auc','epoch':20})
param.append({'task':'binary', 'lr':0.001, 'lambda':0.001, 'metric':'auc','epoch':20})

out_path = '../Input/train/FFM_result'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Training task
ffm_model = xl.create_linear() # Use field-aware factorization machine

for k,fv in enumerate(featrue_version):

    if submission =='y':
        fte = train_path + '/encode/'+fv+'test_fullFFM.txt'
    else:
        fte = train_path + '/encode/'+fv+'test_sample10wFFM.txt'  

    for i,pa in enumerate(param):

        # Prediction task
        ffm_model.setTest(fte)  # Test data
        ffm_model.setSigmoid()  # Convert output to 0-1

        if submission =='y':
            if subparam == i:
                ffm_model.predict(out_path+'/'+fv+'pa'+str(i+1)+'_model.out',
                                out_path+'/'+fv+'pa'+str(i+1)+'_submission_y.txt')
                
                fy = out_path+'/'+fv+'pa'+str(i+1)+'_submission_y.txt'
                y_pred=pd.read_csv(fy,header=None)
                y_pred.columns = ['click']

                fid="../Input/test/summission_id.csv"
                y_id=pd.read_csv(fid)

                subm=pd.concat([y_id['id'],y_pred['click']],axis=1)

                fsub=out_path +'/fm_'+fv+'pa'+str(i+1)+ "_submission.csv"
                subm.to_csv(fsub,mode='w',index=False,header=True)

        else:
            # Start to predict
            # The output result will be stored in output.txt
            ffm_model.predict(out_path+'/'+fv+'pa'+str(i+1)+'_model.out',
                            out_path+'/'+fv+'pa'+str(i+1)+'_output.txt')
                        
