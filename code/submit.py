import pandas as pd
from sklearn.preprocessing import minmax_scale
def gen_submission(sample_file, predict_file, submit_file):
    sample = pd.read_csv(sample_file)
    prediction = pd.read_csv(predict_file)
    print('sample head:',sample.head())
    print('prediction head:',prediction.head())
    sample['click'] = minmax_scale(prediction['1'],feature_range=(0.3,0.7))
    print('after merge:',sample.head())
    sample.to_csv(submit_file,index=False)

sample_file = 'data/sampleSubmission'
predict_file = 'data/predict_prob.csv'
submit_file = 'data/submission_1.csv'
gen_submission(sample_file = sample_file, predict_file = predict_file, submit_file = submit_file)