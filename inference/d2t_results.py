from analysis import D2TStat

d2t_stat = D2TStat('D2T/BAGEL/sescore.pkl')

# Set valid metrics
valid_metrics = [
    #   'rouge1_f',
    #   'rouge2_f',
    #   'rougel_f',
    #   'bert_score_f',
    #   'mover_score',
    #   'prism_avg',
    # 'contra_score',
    'sescore',
    # 'bart_score_avg_f',
    # 'bart_score_para_avg_f',
    # 'bart_score_para_avg_f_de'
]

# The first argument is human metric while the latter is a list of metrics considered.
d2t_stat.evaluate_text('informativeness', valid_metrics)
