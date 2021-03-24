import matplotlib.pyplot as plt
import netgraph
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
import numpy as np
import pandas as pd
import scipy.stats as ss



#############################################
# After model is done, experiment with different networkx plot types
#############################################

# Read in data
df = pd.read_csv("final_estimated_DAG.csv")

# Drop first column
df.drop(df.columns[0], axis=1, inplace = True)

# Get variable names
var_names = list(df.columns)

# Reconstruct the DAG
final_DAG = from_numpy_matrix(df.to_numpy(), create_using = nx.DiGraph)
final_DAG = nx.relabel_nodes(final_DAG, dict(zip(list(range(len(var_names))), var_names)))

# Remove isolates
final_DAG.remove_nodes_from(list(nx.isolates(final_DAG)))

# nx.draw(final_DAG, node_color="lightcoral", node_size=75, font_size=3, width = 0.5, arrowsize=4, with_labels=True, pos=nx.spring_layout(final_DAG))
# plt.draw()
# plt.savefig("DAG_plot.png", format="PNG", dpi=500)
# plt.clf()





#############################################
# Experiment with Netgraph (interactive, drag-able nodes)
#############################################
new_vars = list(final_DAG.nodes)

fig, ax = plt.subplots(1, 1)
ax.set(xlim=[-3, 3], ylim=[-3, 3])

interactive_plot = netgraph.InteractiveGraph(final_DAG, node_labels=dict(zip(new_vars, list(new_vars))), node_size=25, font_size=35, edge_width = 4, ax=ax)
plt.show()





#############################################
# Read in data and explore correlations
#############################################


# Read in the data
df = pd.read_csv("data_5000samples.csv")


def cramers_corrected_stat(x,y):
    """
    Calculate Cramers V statistic for categorial-categorial association
    """
    result=-1
    if len(x.value_counts())==1 :
        print("First variable is constant")
    elif len(y.value_counts())==1:
        print("Second variable is constant")
    else:
        conf_matrix=pd.crosstab(x, y)

        if conf_matrix.shape[0]==2:
            correct=False
        else:
            correct=True

        chi2 = ss.chi2_contingency(conf_matrix, correction=correct)[0]

        n = sum(conf_matrix.sum())
        phi2 = chi2/n
        r,k = conf_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        result=np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    return round(result, 6)



# Look at some random raw correlations
cramers_corrected_stat(df['MA'], df['BQ'])
cramers_corrected_stat(df['SH'], df['RC'])
cramers_corrected_stat(df['B'], df['MF'])



# Examine the HF <- OH -> JP neighborhood
cramers_corrected_stat(df['HF'], df['OH'])
cramers_corrected_stat(df['OH'], df['JP'])
cramers_corrected_stat(df['HF'], df['JP'])

# But if you hold OH constant, is there still a weak association between HF and JP?
df2 = df[df['OH'] == 0]
cramers_corrected_stat(df2['HF'], df2['JP']) # Correlation vanishes! Good!
df2 = df[df['OH'] == 1]
cramers_corrected_stat(df2['HF'], df2['JP']) # Correlation vanishes! Good!



# Now examine the HX - BQ - MA neighborhood a bit closer
cramers_corrected_stat(df['MA'], df['HX'])
cramers_corrected_stat(df['HX'], df['BQ'])
cramers_corrected_stat(df['MA'], df['BQ'])

# What if we hold MA constant
df2 = df[df['MA'] == 0]
cramers_corrected_stat(df2['HX'], df2['BQ']) # This makes sense, there is an independent association between HX and BQ
