styles = {
    "Experiment" : {'style':'k-','zorder':100,'lw':0.7},
    "polynomial rudder" : {'style':'-','color':'red'},
    "semiempirical rudder" : {'style':'b-','label':"semi-empirical rudder"},
    "measured rudder" : {'style':'g-','zorder':-100,'alpha':0.5},
}

for key,values in styles.items():
    if not 'label' in values:
        styles[key]['label'] = key[0].lower() + key[1:]