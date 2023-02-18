# dd_earthquake
This repository is module for Richter's Predictor: Modeling Earthquake Damage competition in DrivenData.

ref:https://www.drivendata.org/competitions/57/nepal-earthquake/page/134/
# Score record
## Bench mark: 0.5815
## Used simply useful columns: 0.6768
- geo_level_1_id
- height_percentage
- has_superstructure_adobe_mud
- has_superstructure_mud_mortar_stone
- has_super_structure_rc_non_engineered
- has_super_structure_timber
- foundation_type
- roof_type
- ground_floor_type
## add level_2 and 3: 0.7280
## Select used columns from bar plot: 0.7173
- dropped columns
  - age
  - plan_configuration
  - has_secondary_use_institution
  - has_secondary_use_school
  - has_secondary_use_industry
  - has_secondary_use_health_post
  - has_secondary_use_gov_office
  - has_secondary_use_use_police
  - has_secondary_use_other
## Used Light GBM model with some preprocessing: 0.7424