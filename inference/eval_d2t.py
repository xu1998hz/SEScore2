from analysis import D2TStat
from utils import *

# tot_data = read_pickle('D2T/BAGEL/final_p.pkl')
# contra_data = read_pickle('D2T/BAGEL/all.pkl')
# for index, sub_dict in tot_data.items():
#     tot_data[index]['scores']['contra_score'] = contra_data[index]['scores']['contra_score']

# save_pickle(tot_data, 'D2T/final_p.pkl')

d2t_stat = D2TStat('D2T/final_p.pkl')

# Set valid metrics
valid_metrics = [
      'rouge1_f',
      'rouge2_f',
      'rougel_f',
      'bert_score_f',
      'mover_score',
      'prism_avg',
     'bart_score_avg_f',
     'contra_score',
]

# The first argument is human metric while the latter is a list of metrics considered.
d2t_stat.evaluate_text('naturalness', valid_metrics)
m1 = 'contra_score'
m2 = 'bart_score_avg_f'
result = d2t_stat.sig_test_two(m1, m2, 'naturalness')

if result == 1:
    print(f'{m1} is significantly better than {m2}')
elif result == -1:
    print(f'{m2} is significantly better than {m1}')
else:
    print('cannot decide')
