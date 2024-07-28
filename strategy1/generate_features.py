from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition.alloy import WenAlloys
import pandas as pd
import copy

if __name__ == '__main__':
    df = pd.read_excel("./strategies/Virtual_samples_100000.xlsx")
    df_all = copy.copy(df)
    df2 = StrToComposition(reduce=True, target_col_id='composition_obj').featurize_dataframe(df, 'formula')

    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              WenAlloys()])
    feature_labels = feature_calculators.feature_labels()
    data_2 = feature_calculators.featurize_dataframe(df2, col_id='composition_obj');
    # print(feature_labels)
    # Generated 163 features
    print('Generated %d features' % len(feature_labels))
    best_features_HV = ['MagpieData avg_dev CovalentRadius', 'MagpieData mean Electronegativity',
                        'MagpieData avg_dev Electronegativity', 'MagpieData mean NpValence',
                        'MagpieData mean NUnfilled', 'MagpieData avg_dev SpaceGroupNumber',
                        'Mean cohesive energy', 'Shear modulus strength model']
    best_features_elongation = ['MagpieData range MendeleevNumber', 'MagpieData avg_dev MendeleevNumber',
                                'MagpieData avg_dev MeltingT', 'MagpieData mean NValence',
                                'MagpieData mean NsUnfilled', 'MagpieData maximum NUnfilled',
                                'MagpieData maximum GSvolume_pa', 'MagpieData avg_dev GSvolume_pa',
                                'MagpieData minimum SpaceGroupNumber', 'MagpieData mode SpaceGroupNumber',
                                'Lambda entropy', 'Electronegativity local mismatch']
    best_features_HV.extend(best_features_elongation)
    print(best_features_HV)
    data_2 = data_2[best_features_HV]
    data_2['formula'] = df_all['formula'].values
    data_2.to_excel("./strategies/Virture_samples_Feature_100000.xlsx", index=False)
