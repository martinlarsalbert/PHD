styles = {
    "VCT":{'style':'k.','zorder':100,'lw':0.7,'label':'VCT'},
    "Experiment" : {'style':'k-','zorder':100,'lw':0.7},
    "polynomial rudder" : {'style':'-','color':'red'},
    "semiempirical rudder" : {'style':'-','color':'#0000ff','label':"semi-empirical rudder"},
    "measured rudder" : {'style':'-','color':'#00ff00','zorder':-100,'alpha':0.5},
    "MMG_rudder": {'style':'m-','label':'MMG rudder'},
}

for key,values in styles.items():
    if not 'label' in values:
        styles[key]['label'] = key[0].lower() + key[1:]