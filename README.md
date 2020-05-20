## BERT-INT

This is code and datasets for BERT-INT

### Dependencies

- Python 3 (tested on 3.6.9)
- Pytorch (tested on 1.1.0)
- [transformers](https://github.com/huggingface/transformers) (tested on 2.1.1)
- Numpy

### How to Run

The model runs in two steps:

#### 1. Fine-tune Basic BERT Unit

To fine-tune the Basic BERT Unit, use: 

```shell
cd basic_bert_unit/
python main.py
```

Note that `basic_bert_unit/Param.py` is the config file.

The obtained Basic BERT Unit and some other data will be stored in:  `../Save_model`

#### 2. Run BERT-based Interaction Model

(Note that when running the BERT-based Interaction model, the parameters of the Basic BERT Unit model will **be fixed**.)

To extract the similarity features and run the BERT-base Interaction Model, use:

```shell
cd ../interaction_model/
python clean_attribute_data.py
python get_entity_embedding.py
python get_attributeValue_embedding.py
python get_neighView_and_desView_interaction_feature.py
python get_attributeView_interaction_feature.py
python interaction_model.py
```

Or directly use:

```shell
cd ../interaction_model/
bash run.sh
```

Note that `interaction_model/Param.py` is the config file.

### Dataset

**Description data**

- `data/dbp15k/2016-10-des_dict`: A dictionary storing entity descriptions, which can be loaded by pickle.load()

The description of the entity is extracted from **DBpedia**(<https://wiki.dbpedia.org/downloads-2016-10>)

**DBP15K**

Initial datasets are from **JAPE**(<https://github.com/nju-websoft/JAPE>).

There are three cross-lingual datasets in folder `data/dbp15k/` , take the dataset DBP15K(ZH-EN) as an example, the folder `data/dbp15k/zh_en` contains:

- ent_ids_1: entity ids and entities in source KG (ZH)
- ent_ids_2: entity ids and entities in target KG (EN)
- ref_pairs: entity links encoded by ids (Test Set)
- sup_pairs: entity links encoded by ids (Train Set)
- rel_ids_1: relation ids and relations in source KG (ZH)
- rel_ids_2: relation ids and relations in target KG (EN)
- triples_1: relation triples encoded by ids in source KG (ZH)
- triples_2: relation triples encoded by ids in target KG (EN)
- zh_att_triples: attribute triples of source KG (ZH)
- en_att_triples: attribute triples of target KG (EN)

