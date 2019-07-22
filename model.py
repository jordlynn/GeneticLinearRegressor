from gplearn.genetic import SymbolicRegressor
from sklearn.utils.random import check_random_state
import numeraidata

loadedData = numeraidata.numeraiData()
print("done init")

rng = check_random_state(0)

est_gp = SymbolicRegressor(population_size=2000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)

est_gp.fit(loadedData.df[loadedData.feature_names], loadedData.df[loadedData.target_name])
print(est_gp._program)

def numerai_score(y_true, y_pred):
     rank_pred = y_pred.groupby(eras).apply(lambda x: x.rank(pct=True, method="first"))
     return numpy.corrcoef(y_true, rank_pred)[0, 1]