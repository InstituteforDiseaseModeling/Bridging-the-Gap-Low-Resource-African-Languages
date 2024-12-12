# For HuggingFace version

import pandas as pd
import os

# Create output dirs
os.makedirs('parquet_ready_release/mmlu_clinical_knowledge', exist_ok=True)
os.makedirs('parquet_ready_release/mmlu_college_medicine', exist_ok=True)
os.makedirs('parquet_ready_release/mmlu_virology', exist_ok=True)
os.makedirs('parquet_ready_release/winogrande', exist_ok=True)

mmlu_dir = 'evaluation_benchmarks_afr_release/mmlu_cm_ck_vir'
for filename in sorted(os.listdir(mmlu_dir)):
    # Skip English files
    if any(filename.endswith(x) for x in ['_dev.csv', '_test.csv', '_val.csv']):
        continue

    # Add header to MMLU
    mmlu_df = pd.read_csv(os.path.join(mmlu_dir, filename), header=None, encoding='utf-8')
    mmlu_df.columns = ['Question', 'OptionA', 'OptionB', 'OptionC', 'OptionD', 'Answer']

    # Determine output directory and check length
    if 'clinical' in filename:
        output_dir = 'parquet_ready_release/mmlu_clinical_knowledge'
        if 'dev' in filename:
            assert mmlu_df.shape[0] == 5
        elif 'test' in filename:
            assert mmlu_df.shape[0] == 265
        else:
            assert mmlu_df.shape[0] == 29
    elif 'college' in filename:
        output_dir = 'parquet_ready_release/mmlu_college_medicine'
        if 'dev' in filename:
            assert mmlu_df.shape[0] == 5
        elif 'test' in filename:
            assert mmlu_df.shape[0] == 173
        else:
            assert mmlu_df.shape[0] == 22
    else:
        output_dir = 'parquet_ready_release/mmlu_virology'
        if 'dev' in filename:
            assert mmlu_df.shape[0] == 5
        elif 'test' in filename:
            assert mmlu_df.shape[0] == 166
        else:
            assert mmlu_df.shape[0] == 18

    # Save to output directory depending on subject
    mmlu_df.to_csv(os.path.join(output_dir, filename), index=False, encoding='utf-8')

wino_dir = 'evaluation_benchmarks_afr_release/winogrande_s'
output_dir = 'parquet_ready_release/winogrande'
for filename in sorted(os.listdir(wino_dir)):
    # Skip English version
    if filename == 'winogrande.csv':
        continue

    language = filename.split('_')[-1].replace('.csv', '')

    # Drop English columns, rename to match columns across languages
    wino_df = pd.read_csv(os.path.join(wino_dir, filename), encoding='utf-8')
    wino_df.drop(wino_df.iloc[:, 1:4], axis=1, inplace=True)
    wino_df.columns = ['qID', 'Sentence', 'Option1', 'Option2', 'Answer', 'Split']

    # Split by... Split
    wino_df_dev = wino_df[wino_df['Split'] == 'dev'].drop(columns=['Split'])
    wino_df_test = wino_df[wino_df['Split'] == 'test'].drop(columns=['Split'])
    wino_df_train = wino_df[wino_df['Split'] == 'train_s'].drop(columns=['Split'])

    # Check lengths
    assert wino_df_dev.shape[0] == 1267
    assert wino_df_test.shape[0] == 1767
    assert wino_df_train.shape[0] == 640

    # Save with MMLU naming convention
    wino_df_dev.to_csv(os.path.join(output_dir, f'winogrande_dev_{language}.csv'), index=False, encoding='utf-8')
    wino_df_test.to_csv(os.path.join(output_dir, f'winogrande_test_{language}.csv'), index=False, encoding='utf-8')
    wino_df_train.to_csv(os.path.join(output_dir, f'winogrande_train_s_{language}.csv'), index=False, encoding='utf-8')
