#!/bin/sh
python -u clean_attribute_data.py
python -u get_entity_embedding.py
python -u get_attributeValue_embedding.py
python -u get_neighView_and_desView_interaction_feature.py
python -u get_attributeView_interaction_feature.py
python -u interaction_model.py
