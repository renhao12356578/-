import evaluation_utils as eval_utils
import json
import statistics as stats
import range_query as rq
import learn_from_query


def gen_report(act, est_results, saved_path):
    f = open(saved_path + "/report.md", "w")
    f.write("| name | p50 | p80 | p90 | p99 |\n")
    f.write("| --- | --- | --- | --- | --- |\n")
    for name in est_results:
        est = est_results[name]
        eval_utils.draw_act_est_figure(name, act, est)
        p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(act, est)
        f.write("| %s | %.2f | %.2f | %.2f | %.2f |\n" % (name, p50, p80, p90, p99))

    f.write("\n")
    for name in est_results:
        f.write("![%s](%s.png)\n\n" % (name, name))
    f.close()

    with open('./eval/results.json', 'w') as outfile:
        est_results['act'] = act
        json.dump(est_results, outfile)


if __name__ == '__main__':
    stats_json_file = 'data/title_stats.json'
    train_json_file = 'data/query_train_18000.json'
    test_json_file = 'data/validation_2000.json'
    columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']

    table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    with open(test_json_file, 'r') as f:
        test_data = json.load(f)

    est_avi, est_ebo, est_min_sel, act = [], [], [], []
    for item in test_data:
        range_query = rq.ParsedRangeQuery.parse_range_query(item['query'])
        est_avi.append(stats.AVIEstimator.estimate(range_query, table_stats) * table_stats.row_count)
        est_ebo.append(stats.ExpBackoffEstimator.estimate(range_query, table_stats) * table_stats.row_count)
        est_min_sel.append(stats.MinSelEstimator.estimate(range_query, table_stats) * table_stats.row_count)
        act.append(item['act_rows'])

    model_saved_path = './trained_model'
    train_dataset, test_dataset = learn_from_query.load_data(train_data, test_data, table_stats, columns)
    # _, _, est_AI1, _ = learn_from_query.est_AI1(train_dataset, test_dataset, model_saved_path)
    '''
    _, _, est_AI2, _ = learn_from_query.est_AI2(train_dataset, test_dataset, model_saved_path)
    _, _, est_AI3, _ = learn_from_query.est_AI3(train_dataset, test_dataset, model_saved_path)
    '''
    _, _, est_AI4, _ = learn_from_query.est_AI4(train_dataset, test_dataset, model_saved_path)
    '''
    _, _, est_AI5, _ = learn_from_query.est_AI5(train_dataset, test_dataset, model_saved_path)
    _, _, est_AI6, _ = learn_from_query.est_AI6(train_dataset, test_dataset, model_saved_path)
    _, _, est_AI7, _ = learn_from_query.est_AI7(train_dataset, test_dataset, model_saved_path)
    '''

    results_saved_path = './eval'
    gen_report(act, {
        "avi": est_avi,
        "ebo": est_ebo,
        "min_sel": est_min_sel,
        # "LogisticRegression": est_AI1
        #"KNN": est_AI3,
        "SVM": est_AI4,
        #"RandomForest": est_AI2,
        #"LightBGM": est_AI5,
        #"XGBoost": est_AI6,
        #"MLP": est_AI7
    }, results_saved_path)
