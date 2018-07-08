
import xlearn as xl
import os
import pandas as pd

featrue_version = 'v4'

subparam = 0
model ='FFM'
parav ='parav3'
pai = 1

in_path = "/media/leikun/programs/AI100/AI100-Final/Input/train/" + featrue_version+ "_" + "data"
out_path = in_path+ '/'+model+'_result' + '/' + parav

# Training task
if model == 'LR':
    ffm_model = xl.create_linear() # Use field-aware factorization machine

if model == 'FFM':
    ffm_model = xl.create_ffm() 

if model == 'FM':
    ffm_model = xl.create_fm() 

# On-disk training
ffm_model.setOnDisk()

fte = in_path  +'/test'+featrue_version+'.ffm'

# Prediction task
ffm_model.setTest(fte)  # Test data
ffm_model.setSigmoid()  # Convert output to 0-1

ffm_model.predict(out_path+'/'+'pa'+str(pai)+'_model.out',
                out_path+'/'+'pa'+str(pai)+'_submission_y.txt')

fy = out_path+'/'+'pa'+str(pai)+'_submission_y.txt'
y_pred=pd.read_csv(fy,header=None)
y_pred.columns = ['click']

fid=in_path + '/label_test'+featrue_version+'.ffm' # test集训练时候的的id顺序
y_pre_id=pd.read_csv(fid,header=None)
y_pre_id.columns = ['id']

subm=pd.concat([y_pre_id['id'],y_pred['click']],axis=1)

fid="/media/leikun/programs/AI100/AI100-Final/Input/test/summission_id.csv"
df_id=pd.read_csv(fid) # 递交的原始文件id顺序
y_id= df_id['id']

# 把输出的预测结果,id按照原来的重新排列
# 设置成“category”数据类型
subm['id'] = subm['id'] .astype('category')
# inplace = True，使 recorder_categories生效
subm['id'] .cat.reorder_categories(y_id, inplace=True)
# inplace = True，使 subm生效
subm.sort_values('id', inplace=True) #

fsub=out_path + '/' + model+'_'+featrue_version+'_pa'+str(pai)+ "_submission.csv"
subm.to_csv(fsub,mode='w',index=False,header=True)

                
