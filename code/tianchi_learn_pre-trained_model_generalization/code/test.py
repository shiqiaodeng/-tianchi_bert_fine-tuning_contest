task_num_classes = {'0': 3, '1': 7, '2': 15}
task_id_to_name = {'0': 'ocnli', '1': 'ocemotion', '2': 'tnews'}
task_lambda = {'0': 1.0, '1': 1.5, '2': 1.5}
task_data = {
    'ocemotion': {'predict': 'OCEMOTION_a.csv',
                  'train': 'OCEMOTION_train1128.csv',
                  'test_b': 'ocemotion_test_B.csv'},
    'ocnli': {'predict': 'OCNLI_a.csv',
              'train': 'OCNLI_train1128.csv',
              'test_b': 'ocnli_test_B.csv'},
    'tnews': {'predict': 'TNEWS_a.csv',
              'train': 'TNEWS_train1128.csv',
              'test_b': 'tnews_test_B.csv'}
}


def load_json(file_path):
    return json.load(open(file_path, 'r', encoding='utf8'))


def get_df(data_dir: str, data_name: str) -> pd.DataFrame:
    data_path = os.path.join(data_dir, data_name)
    df = pd.read_csv(data_path, sep='\t', header=None)
    return df


def preprocess(args: Any, task: str, test_b: bool = False):
    data_name = task_data[task]
    if test_b:
        data_name['predict'] = data_name['test_b']
    train_df = get_df(args.data_dir, data_name['train'])
    pred_df = get_df(args.data_dir, data_name['predict'])
    
    pretrained_model_path = args.tokenizer_dir
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    train_precessed, val_precessed, label2id = convert_df_to_inputs(task, tokenizer, train_df,
                                                                    args.train_val_split_ratio, debug=args.data_debug)
    predict_precessed, = convert_df_to_inputs(task, tokenizer, pred_df, label2id=label2id, debug=args.data_debug)

    data_save_dir = os.path.join(args.data_save_dir, task)
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    print(f'Saving processed {task} data ...')
    cache_data_path = {
        'train': args.data_save_dir + '/' + task + '/' + 'train.pt',
        'val': args.data_save_dir + '/' + task + '/' + 'val.pt',
        'predict': args.data_save_dir + '/' + task + '/' + 'predict.pt'
    }
    json.dump(label2id, open(os.path.join(data_save_dir, 'label2id.json'), 'w'))
    torch.save(train_precessed, cache_data_path['train'])
    torch.save(val_precessed, cache_data_path['val'])
    torch.save(predict_precessed, cache_data_path['predict'])

    return cache_data_path


def convert_label_to_id(targets_series: pd.Series, label2id: Optional[dict] = None) -> tuple:
    # print("convert_label_to_id targets_series: ", targets_series)
    # print("convert_label_to_id label2id: ", label2id)
    labels = np.unique(targets_series.values)
    train = False
    if label2id is None:
        train = True
        label2id = {str(label): i for i, label in enumerate(labels)}
    # print("convert_label_to_id label2id: ", label2id)
    targets_series = targets_series.apply(lambda label: str(label))
    # print("convert_label_to_id targets_series: ", targets_series)
    targets_series = targets_series.apply(lambda label: label2id[label])
    # print("convert_label_to_id targets_series: ", targets_series)
    targets = torch.from_numpy(targets_series.values.astype('int64'))
    # print("convert_label_to_id targets: ", targets)
    outputs = (targets,)
    if train:
        outputs += (label2id,)
    # print("convert_label_to_id train: ", train)
    # print("convert_label_to_id outputs: ", outputs)
    return outputs


def convert_df_to_inputs(task: str, tokenizer: BertTokenizer, df: pd.DataFrame,
                         train_val_split_ratio: Optional[float] = None,
                         label2id: Optional[dict] = None, debug: bool = False) -> tuple:
    inputs = defaultdict(list)
    train = False
    if debug:
        df = df.head(1000)
    df.sample(frac=1, replace=True, random_state=32)
    label2id, train = _iter_row(df, inputs, task, tokenizer, train, train_val_split_ratio, label2id)

    if train_val_split_ratio is not None:
        outputs = train_val_split(inputs, train_val_split_ratio)
    else:
        outputs = (inputs,)

    if train:
        outputs += (label2id,)

    # print("convert_df_to_inputs outputs: ", outputs)
    return outputs


def _iter_row(df, inputs: dict, task: str, tokenizer: BertTokenizer, train: bool,
              train_val_split_ratio: float, label2id: Optional[dict] = None) -> Tuple[dict, bool]:
    targets = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Preprocess {task}'):
        text_a = row[1]
        if task == 'ocnli':
            target_idx = 3
            text_b = row[2]
            output_ids = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True,
                                               return_token_type_ids=True, return_attention_mask=True)
            # print("_iter_row output_ids: ", output_ids)
        else:
            target_idx = 2
            output_ids = tokenizer.encode_plus(text_a, add_special_tokens=True,
                                               return_token_type_ids=True, return_attention_mask=True)
        inputs['input_ids'].append(output_ids['input_ids'])
        inputs['token_type_ids'].append(output_ids['token_type_ids'])
        inputs['attention_mask'].append(output_ids['attention_mask'])

        if train_val_split_ratio is not None:
            targets.append(row[target_idx])
        else:
            targets.append(list(label2id.keys())[0])
    targets_series = pd.Series(targets)
    if label2id is None:
        train = True
        targets, label2id = convert_label_to_id(targets_series)
    else:
        targets, = convert_label_to_id(targets_series, label2id)
    inputs['targets'] = targets
    # print("_iter_row inputs: ", inputs)
    # print("_iter_row label2id: ", label2id)
    # print("_iter_row train: ", train)
    return label2id, train


def train_val_split(inputs, train_val_split_ratio):
    num_val = int(len(inputs['input_ids']) * train_val_split_ratio)
    train_data = {}
    val_data = {}
    for key, tensor in inputs.items():
        train_data[key] = tensor[num_val:]
        val_data[key] = tensor[:num_val]
    outputs = (train_data, val_data)
    return outputs