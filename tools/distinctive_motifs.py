
'''
Extract ranked distinctive motifs ignoring artifacts
'''
from os import path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

colors_map = {
    'perfect': 'green',
    'mixed': 'blue',
    'artifact': 'red',
    'negative': 'orange'
}

def get_sorted_features(feature_importance_path: str):
    with open(feature_importance_path, 'r') as f:
        features = [line.strip().split('\t') for line in f.readlines()]
    features = sorted(features, key=lambda x: float(x[1]), reverse=True)
    return features


def is_artifact(motif: str, values: pd.DataFrame, bio_cond: str, invalid_mix: str, score: float, order: int):
    data = values[['label', 'sample_name', motif]]
    other_max = data.loc[data['label'] == 'other', motif].max()
    bc_min = data.loc[data['label'] == bio_cond, motif].min()
    artifact = bc_min < other_max
    is_perfect = bc_min > other_max
    is_valid_mix = True
    mixed_samples = list(data.loc[(data['label'] == 'other') & (data[motif] >= bc_min), 'sample_name'])
    if artifact:
        print(f'Motif {motif} is artifact, other_max={other_max}, bc_min={bc_min}, importance score={score}, importance order={order}')
    elif is_perfect:
        print(f'Motif {motif} is perfect, other_max={other_max}, bc_min={bc_min}, importance score={score}, importance order={order}')
    else:
        if invalid_mix:
            is_valid_mix = not any(invalid_mix in s for s in mixed_samples)
        if is_valid_mix:
            print(f'Motif {motif} is mixed, other_max={other_max}, bc_min={bc_min}, importance score={score}, importance order={order}, mixes={mixed_samples}')
        else:
            print(f'Motif {motif} has invalid mix, other_max={other_max}, bc_min={bc_min}, importance score={score}, importance order={order}, mixes={mixed_samples}')
        # TODO convert mixed samples to mixed bc inc. count
    return artifact, is_perfect, is_valid_mix, mixed_samples


def generate_heatmap(base_path: str, df: pd.DataFrame, colors, title: str):
    print('Generating heatmap...')

    df.set_index('sample_name', inplace=True)
    map_path = f'{base_path}.svg'
    number_of_samples = df.shape[0]

    map = sns.clustermap(df, cmap="Blues", col_cluster=False, yticklabels=True, col_colors=colors)
    plt.setp(map.ax_heatmap.yaxis.get_majorticklabels(), fontsize=150 / number_of_samples)
    map.ax_heatmap.set_title(title, pad=25, fontsize=14)
    map.savefig(map_path, format='svg', bbox_inches="tight")
    plt.close()


def save_output(base_path: str, data):
    print('Saving results...')
    output_path = f'{base_path}.csv'
    df = pd.DataFrame(data, columns=['motif', 'label', 'is_artifact', 'is_perfect', 'is_valid_mix', 'mixed_samples', 'importance', 'order'])
    df.to_csv(output_path, index=False)


def extract_distinctive_motifs(count: int, epsilon: float, feature_importance_path: str, values_path: str, hits_path: str, invalid_mix: str, min_importance_score: float, output_base_path: str, heatmap_title: str):
    features = get_sorted_features(feature_importance_path)
    values = pd.read_csv(values_path)
    # hits = pd.read_csv(hits_path, index_col=[1,0])
    # TODO check if motif is backed by hits (only log, no filter)
    bio_conds = list(values['label'].unique())
    bio_conds.remove('other')
    bio_cond = bio_conds[0]
    
    total = 0
    last_score = 0
    last_order = 0
    distinctive_motifs = []
    i = 0
    perfect_count = 0
    artifact_count = 0
    invalid_mix_count = 0
    mixed_count = {}
    colors = []
    output = []
    output_label = ''
    for feature in features:
        i += 1
        motif = feature[0]
        score = float(feature[1])
        if score < min_importance_score:
            break
        artifact, is_perfect, is_valid_mix, mixed_samples = is_artifact(motif, values, bio_cond, invalid_mix, score, i)
        if total >= count and score + epsilon < last_score:
            break
        if artifact:
            artifact_count += 1
            colors.append(colors_map['artifact'])
            output_label = 'artifact'
            output.append([motif, output_label, artifact, is_perfect, is_valid_mix, mixed_samples, score, i])
            continue
        if not is_valid_mix:
            invalid_mix_count += 1
            colors.append(colors_map['negative'])
            output_label = 'negative'
            output.append([motif, output_label, artifact, is_perfect, is_valid_mix, mixed_samples, score, i])
            continue
        if is_perfect:
            perfect_count += 1
            colors.append(colors_map['perfect'])
            output_label = 'perfect'
        else:
            colors.append(colors_map['mixed'])
            output_label = 'mixed'
            for sample in mixed_samples:
                sample_count = 0
                try:
                    sample_count = mixed_count[sample]
                except:
                    pass
                mixed_count[sample] = sample_count + 1
        distinctive_motifs.append(motif)
        output.append([motif, output_label, artifact, is_perfect, is_valid_mix, mixed_samples, score, i])
        last_order = i
        total += 1
        if total == count:
            last_score = score
    
    if output_base_path:
        columns = ['sample_name'] + [x[0] for x in features[:i - 1]]
        generate_heatmap(output_base_path, values[columns], colors, heatmap_title)
        save_output(output_base_path, output)
    return distinctive_motifs, last_order, perfect_count, artifact_count, invalid_mix_count, mixed_count


if __name__ == '__main__':
    base_path = '/home/shenson/Deep_Panning/DP_fix_rf/analysis/model_fitting'
    count = 1000
    epsilon = 0
    feature_importance_path = path.join(base_path, 'BP087_77/BP087_77_values_model/best_model/sorted_feature_importance.txt')
    values_path = path.join(base_path, 'BP087_77/BP087_77_values.csv')
    hits_path = path.join(base_path, 'BP087_77/BP087_77_hits.csv')
    invalid_mix = 'BP087_0' # 'naive'
    min_importance_score = 0
    output_base_path = path.join(base_path, 'BP087_77/distinctive_motifs_all')
    heatmap_title = 'BP087_77 Distinctive Motifs (all)'
   
    motifs, last_index, perfect_count, artifact_count, invalid_mix_count, mixed_count = extract_distinctive_motifs(count, epsilon, feature_importance_path, values_path, hits_path, invalid_mix, min_importance_score, output_base_path, heatmap_title)
    print(f'\nDistinctive motifs ({len(motifs)}/{last_index} tested): {motifs}')
    print(f'Perfects count: {perfect_count}')
    print(f'Mixed count: {mixed_count}')
    print(f'Filtered: Artifacts count: {artifact_count}, Invalid mixes count: {invalid_mix_count}')
