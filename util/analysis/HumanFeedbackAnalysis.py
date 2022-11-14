import pandas as pd
from os import listdir
from os.path import join


def process_feedback_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    problem, run_id = df['problem'].tolist()[0], df['run_id'].tolist()[0]
    df['misprediction'] = abs((df['prediction'] - df['feedback'])) / 2
    df_count_mispredictions = df.groupby(['resp_iteration'])['misprediction'].sum().reset_index(name='mispredictions')
    df_count_feedback = df.groupby(['resp_iteration'])['resp_iteration'].count().reset_index(name='n_responses')
    df_avg_time = df.groupby(['resp_iteration'])['duration'].mean().reset_index(name='average_response_time')
    merged = pd.merge(df_count_mispredictions, df_count_feedback, how='inner', on='resp_iteration')
    merged = pd.merge(merged, df_avg_time, how='inner', on='resp_iteration')
    resp_iterations = merged['resp_iteration'].tolist()
    for iteration in range(51):
        if iteration not in resp_iterations:
            merged.loc[len(merged.index)] = [iteration, None, 0, None]
    sorted_df = merged.sort_values(by=['resp_iteration'], ignore_index=True)
    sorted_df['problem'] = problem
    sorted_df['run_id'] = run_id
    sorted_df['total_responses'] = len(df)
    sorted_df['overall_average_response_time'] = sum(df['duration'].tolist()) / len(df['duration'].tolist())
    return sorted_df


def process_survey_files(folder: str):
    dataframes = []
    for file in listdir(folder):
        if file.startswith('survey'):
            run_id = file.replace('survey-', '').replace('.csv', '')
            filename = join(folder, file)
            survey_df = pd.read_csv(filename)
            survey_df['run_id'] = run_id
            try:
                feedback_df = pd.read_csv(join(folder, 'feedback-' + run_id + '.csv'))
            except FileNotFoundError:
                continue
            survey_df['n_feedback'] = len(feedback_df)
            dataframes.append(survey_df)
    pd.concat(dataframes, ignore_index=True).to_csv('D:/Research/ML-PIE/human_survey.csv', index=False)


def process_feedback_files(folder: str):
    dataframes = []
    for file in listdir(folder):
        if file.startswith("feedback"):
            filename = join(folder, file)
            dataframes.append(process_feedback_dataframe(pd.read_csv(filename)))
    pd.concat(dataframes, ignore_index=True).to_csv('D:/Research/ML-PIE/human_feedback.csv', index=False)


if __name__ == '__main__':
    directory = 'D:/Research/ML-PIE/humanresults_filtered'
    process_feedback_files(directory)
    process_survey_files(directory)
