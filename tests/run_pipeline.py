"""Smoke test: profile -> fixed plan -> apply -> re-profile on the messy sample."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.loaders import load_dataframe
from app.profiler import profile_dataframe
from app.cleaner import apply_plan

df, meta = load_dataframe('sample_data/messy_employees.csv')
p1 = profile_dataframe(df)
print('SCORE BEFORE', p1['score']['score'], p1['score']['grade'], '| issues', len(p1['issues']))
acts = [
 {'id':'a1','action':'fix_encoding','column':' Full Name '},
 {'id':'a2','action':'strip_whitespace','column':' Full Name '},
 {'id':'a3','action':'normalize_case','column':' Full Name ','params':{'case':'title'}},
 {'id':'a4','action':'replace_disguised_nulls','column':'Age'},
 {'id':'a5','action':'parse_numeric','column':'Age'},
 {'id':'a6','action':'replace_values_with_null','column':'Age','params':{'values':['999']}},
 {'id':'a7','action':'replace_values_with_null','column':'Age','params':{'condition':'gt','value':120}},
 {'id':'a8','action':'remove_rows_where','column':'Age','params':{'condition':'lt','value':0}},
 {'id':'a9','action':'fill_missing','column':'Age','params':{'strategy':'median'}},
 {'id':'b1','action':'parse_numeric','column':'Salary (USD)'},
 {'id':'b2','action':'handle_outliers','column':'Salary (USD)','params':{'method':'iqr','treatment':'clip','factor':1.5}},
 {'id':'b3','action':'fill_missing','column':'Salary (USD)','params':{'strategy':'median'}},
 {'id':'c1','action':'parse_dates','column':'Join Date','params':{'dayfirst':False}},
 {'id':'c2','action':'remove_rows_where','column':'Join Date','params':{'condition':'gt','value':'2027'}},
 {'id':'c3','action':'replace_values_with_null','column':'Join Date','params':{'condition':'lt','value':'1901-01-01'}},
 {'id':'d1','action':'strip_whitespace','column':'Department'},
 {'id':'d2','action':'normalize_categories','column':'Department'},
 {'id':'d3','action':'normalize_categories','column':'City','params':{'mapping':[{'from':'Bengaluru','to':'Bangalore'}]}},
 {'id':'d4','action':'normalize_categories','column':'City'},
 {'id':'e1','action':'normalize_boolean','column':'Is Active?'},
 {'id':'e2','action':'standardize_email','column':'Email'},
 {'id':'e3','action':'standardize_phone','column':'Phone#'},
 {'id':'e4','action':'parse_numeric','column':'Rating %'},
 {'id':'e5','action':'clip_values','column':'Rating %','params':{'lower':0,'upper':100}},
 {'id':'f1','action':'drop_columns','params':{'columns':['Unnamed: 14','Country','Department.1']}},
 {'id':'f2','action':'drop_empty_rows','params':{'threshold':0.6}},
 {'id':'f3','action':'drop_rows_with_missing','params':{'columns':['Employee ID']}},
 {'id':'f4','action':'drop_duplicate_rows','params':{'keep':'first'}},
 {'id':'f5','action':'drop_duplicate_rows','params':{'subset':['Employee ID']}},
 {'id':'f6','action':'fix_encoding','column':'Notes'},
 {'id':'f7','action':'replace_disguised_nulls','column':'Notes'},
 {'id':'f8','action':'fill_missing','column':'Notes','params':{'strategy':'constant','value':'none'}},
 {'id':'g1','action':'standardize_column_names'},
]
out, log = apply_plan(df, acts)
bad = [e for e in log if e['status'] != 'applied']
for e in log:
    print(f"{e['status']:8} {e['action']:26} {str(e['column']):14} rows {e['rows_before']}->{e.get('rows_after')} changed={e.get('cells_changed')} nulls {e.get('nulls_before')}->{e.get('nulls_after')} | {e['message'][:80]}")
assert not bad, bad
print(out.dtypes.to_dict())
p2 = profile_dataframe(out)
print('SCORE AFTER', p2['score'])
for i in p2['issues']:
    print(f"[{i['severity']:6}] {i['column'] or 'DATASET':15} {i['type']:28} {i['title']}")
json.dumps(p2)  # must be serialisable
print('OK')
