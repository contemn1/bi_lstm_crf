import os
import codecs
import tensorflow as tf
import sklearn


def prediction_step_pos(sess, dataset, dataset_type, model, transition_params_trained,
                        stats_graph_folder, epoch_number, parameters, dataset_filepaths):
    if dataset_type == 'deploy':
        print('Predict labels for the {0} set'.format(dataset_type))
    else:
        print('Evaluate model on the {0} set'.format(dataset_type))
    all_predictions = []
    all_y_true = []
    output_filepath = os.path.join(stats_graph_folder,
                                   '{1:03d}_{0}.txt'.format(dataset_type, epoch_number))
    output_file = codecs.open(output_filepath, 'w', 'UTF-8')
    original_conll_file = codecs.open(dataset_filepaths[dataset_type], 'r', 'UTF-8')

    for i in range(len(dataset.token_indices[dataset_type])):
        feed_dict = {
            model.input_token_indices: dataset.token_indices[dataset_type][i],
            model.input_token_character_indices: dataset.character_indices_padded[dataset_type][i],
            model.input_token_lengths: dataset.token_lengths[dataset_type][i],
            model.input_label_indices_vector: dataset.label_vector_indices[dataset_type][i],
            model.dropout_keep_prob: 1.
        }
        unary_scores, predictions = sess.run([model.unary_scores, model.predictions], feed_dict)
        if parameters['use_crf']:
            predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params_trained)
            predictions = predictions[1:-1]
        else:
            predictions = predictions.tolist()

        assert (len(predictions) == len(dataset.tokens[dataset_type][i]))
        output_string = ''
        prediction_labels = [dataset.index_to_label[prediction] for prediction in predictions]
        gold_labels = dataset.labels[dataset_type][i]

        all_predictions.extend(predictions)
        all_y_true.extend(dataset.label_indices[dataset_type][i])

    print(sklearn.metrics.classification_report(all_y_true, all_predictions, digits=4,
                                                labels=dataset.label_indices,
                                                target_names=dataset.unique_labels))

    return all_predictions, all_y_true, output_filepath


def predict_labels_pos(sess, model, transition_params_trained, parameters, dataset, epoch_number,
                       stats_graph_folder, dataset_filepaths):
        # Predict labels using trained model
    y_pred = {}
    y_true = {}
    output_filepaths = {}
    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        if dataset_type not in dataset_filepaths.keys():
            continue
        prediction_output = prediction_step_pos(sess, dataset, dataset_type, model,
                                                transition_params_trained, stats_graph_folder,
                                                epoch_number, parameters, dataset_filepaths)
        y_pred[dataset_type], y_true[dataset_type], output_filepaths[
                dataset_type] = prediction_output

    return y_pred, y_true, output_filepaths
