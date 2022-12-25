import pandas as pd
import json

file_name = '/Users/challenge-2020/evaluation/human-evaluation/results/en/english_humeval_data_all_teams.json'
out_name = '/Users/challenge-2020/submissions/rdf2text/en/Amazon_AI_(Shanghai)/primary.en'
ref_name = '/Users/challenge-2020/evaluation/references/references-en.json'

out_lines = open(out_name, 'r').readlines()
out_lines = [line[:-1] for line in out_lines]
human_score_dict = json.load(open(file_name))
ref_dict = json.load(open(ref_name))

sys_score_output_dict = {}
for human_ele_dict in human_score_dict:
    if human_ele_dict['submission_id'] not in sys_score_output_dict:
        sys_score_output_dict[human_ele_dict['submission_id']] = {}
        sys_score_output_dict[human_ele_dict['submission_id']]['Correctness'] = {}
        sys_score_output_dict[human_ele_dict['submission_id']]['DataCoverage'] = {}
        sys_score_output_dict[human_ele_dict['submission_id']]['Fluency'] = {}
        sys_score_output_dict[human_ele_dict['submission_id']]['Relevance'] = {}
        sys_score_output_dict[human_ele_dict['submission_id']]['TextStructure'] = {}
        sys_score_output_dict[human_ele_dict['submission_id']]['Output'] = {}
        sys_score_output_dict[human_ele_dict['submission_id']]['Ref'] = {}
        # sys_score_output_dict[human_ele_dict['submission_id']]['Out_Refs_0'] = {}
        # sys_score_output_dict[human_ele_dict['submission_id']]['Out_Refs_1'] = {}
        # sys_score_output_dict[human_ele_dict['submission_id']]['Out_Refs_2'] = {}
        # sys_score_output_dict[human_ele_dict['submission_id']]['Out_Refs_3'] = {}

    if human_ele_dict['sample_id'] != '1124':
        sys_score_output_dict[human_ele_dict['submission_id']]['Correctness'][human_ele_dict['sample_id']] = float(human_ele_dict['Correctness'])
        sys_score_output_dict[human_ele_dict['submission_id']]['DataCoverage'][human_ele_dict['sample_id']] = float(human_ele_dict['DataCoverage'])
        sys_score_output_dict[human_ele_dict['submission_id']]['Fluency'][human_ele_dict['sample_id']] = float(human_ele_dict['Fluency'])
        sys_score_output_dict[human_ele_dict['submission_id']]['Relevance'][human_ele_dict['sample_id']] = float(human_ele_dict['Relevance'])
        sys_score_output_dict[human_ele_dict['submission_id']]['TextStructure'][human_ele_dict['sample_id']] = float(human_ele_dict['TextStructure'])
        # save all pairs of outputs and refs
        sys_score_output_dict[human_ele_dict['submission_id']]['Output'][human_ele_dict['sample_id']] = out_lines[int(human_ele_dict['sample_id'])-1]

        ref_ele_dict = ref_dict['entries'][int(human_ele_dict['sample_id'])-1][human_ele_dict['sample_id']]['lexicalisations'][0]
        sys_score_output_dict[human_ele_dict['submission_id']]['Ref'][human_ele_dict['sample_id']] = ref_ele_dict['lex']
        # sys_score_output_dict[human_ele_dict['submission_id']]['Out_Refs_0'][human_ele_dict['sample_id']] = []
        # sys_score_output_dict[human_ele_dict['submission_id']]['Out_Refs_1'][human_ele_dict['sample_id']] = []
        # sys_score_output_dict[human_ele_dict['submission_id']]['Out_Refs_2'][human_ele_dict['sample_id']] = []
        # sys_score_output_dict[human_ele_dict['submission_id']]['Out_Refs_3'][human_ele_dict['sample_id']] = []

# sanity check for the length of each system outputs
# common_segs_set = set(sys_score_output_dict['Amazon_AI_(Shanghai)']['Output'].keys())
# for sys in sys_score_output_dict:
#     common_segs_set = common_segs_set.intersection(set(sys_score_output_dict[sys]['Output'].keys()))
#     print(len(sys_score_output_dict[sys]['Output'].keys()))

with open(f'webnlg2020.json', 'w') as f:
    json.dump(sys_score_output_dict, f)

print("File is saved!")
