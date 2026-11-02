import os
import sys
import unittest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

from configs import config_microbial_dataset as ds_cfg  # noqa: E402
from configs import config_microbial_models as model_cfg  # noqa: E402
from configs import config_microbial_AD as ad_cfg  # noqa: E402


class TestShannonDatasetConfig(unittest.TestCase):
    def test_differs_from_faith_pd_config_only_in_target(self):
        base = ds_cfg.config_novel_alpha_faith_pd
        new = ds_cfg.config_novel_alpha_shannon
        self.assertEqual(new["dataset_name"], "microbial_novel_alpha_shannon")
        self.assertEqual(new["microbial_features"], ["div_alpha_shannon"])
        self.assertEqual(set(new), set(base))
        for key in set(base) - {"dataset_name", "microbial_features"}:
            self.assertEqual(new[key], base[key], key)


ID57_SELECTION = {
    "dataset": "microbial_novel_alpha_faith_pd",
    "use_rnn": True,
    "input_sig": False,
    "val_predict_for_t": 366.0 / 1162,
    "use_only_dyn_ft_as_input": "lambda t,x: (torch.rand(x) < (min(t,5000)-2000)/6000)",
}


class TestShannonModelConfig(unittest.TestCase):
    def test_single_config_equals_published_model_except_target(self):
        reference = [p for p in model_cfg.param_list_microbial_novel_alpha_div2
                     if all(p[k] == v for k, v in ID57_SELECTION.items())]
        self.assertEqual(len(reference), 1)
        self.assertIs(reference[0], model_cfg.param_list_microbial_novel_alpha_div2[56])  # id-57: model ids start at 1
        expected = dict(reference[0])
        expected["dataset"] = "microbial_novel_alpha_shannon"
        expected["saved_models_path"] = model_cfg.microbial_models_path_novel_alpha_div_shannon
        self.assertEqual(model_cfg.param_list_microbial_novel_alpha_div_shannon, [expected])

    def test_saved_models_path_and_overview(self):
        path = model_cfg.microbial_models_path_novel_alpha_div_shannon
        self.assertTrue(path.endswith("saved_models_microbial_novel_alpha_div_shannon/"))
        overview = model_cfg.overview_dict_microbial_novel_alpha_div_shannon
        self.assertEqual((overview["ids_from"], overview["ids_to"]), (1, 1))
        self.assertEqual(overview["path"], path)

    def test_benchmark_config_differs_only_in_dataset_epochs_pretrain_and_path(self):
        [shannon] = model_cfg.param_list_microbial_novel_alpha_div_shannon
        expected = dict(shannon)
        expected.update({"dataset": "microbial_novel_alpha_faith_pd", "epochs": 5, "pre-train": 0,
                         "saved_models_path": model_cfg.microbial_models_path_x_bench})
        self.assertEqual(model_cfg.param_list_microbial_novel_alpha_div_bench, [expected])


class TestShannonADConfig(unittest.TestCase):
    def test_path_and_ids_point_at_the_shannon_model(self):
        self.assertEqual(ad_cfg.AD_microbial_novel_alpha_div_shannon,
                         model_cfg.microbial_models_path_novel_alpha_div_shannon)
        self.assertEqual(ad_cfg.AD_microbial_novel_alpha_div_shannon_ids, [1])


class TestRetrainingId57Config(unittest.TestCase):
    def test_single_config_equals_published_model(self):
        self.assertEqual(model_cfg.param_list_microbial_novel_alpha_div2_id57,
                         [model_cfg.param_list_microbial_novel_alpha_div2[56]])

    def test_overview_covers_id57_only(self):
        overview = model_cfg.overview_dict_microbial_novel_alpha_div2_id57
        self.assertEqual((overview["ids_from"], overview["ids_to"]), (57, 57))
        self.assertEqual(overview["path"], model_cfg.microbial_models_path_novel_alpha_div2)


class TestCountryModelConfig(unittest.TestCase):
    def test_dataset_adds_country_as_last_covariate(self):
        base = ds_cfg.config_novel_alpha_faith_pd
        new = ds_cfg.config_novel_alpha_faith_pd_w_geo
        self.assertEqual(new["dynamic_features"], base["dynamic_features"] + ["geo_location_name"])
        for key in set(base) - {"dataset_name", "dynamic_features"}:
            self.assertEqual(new[key], base[key], key)

    def test_single_config_equals_published_model_except_dataset_and_path(self):
        expected = dict(model_cfg.param_list_microbial_novel_alpha_div2[56])
        expected["dataset"] = "microbial_novel_alpha_faith_pd_w_geo"
        expected["saved_models_path"] = model_cfg.microbial_models_path_novel_alpha_div_country
        self.assertEqual(model_cfg.param_list_microbial_novel_alpha_div_country, [expected])

    def test_saved_models_path_overview_and_ad_config(self):
        path = model_cfg.microbial_models_path_novel_alpha_div_country
        self.assertTrue(path.endswith("saved_models_microbial_novel_alpha_div_country/"))
        overview = model_cfg.overview_dict_microbial_novel_alpha_div_country
        self.assertEqual((overview["ids_from"], overview["ids_to"]), (1, 1))
        self.assertEqual(overview["path"], path)
        self.assertEqual(ad_cfg.AD_microbial_novel_alpha_div_country, path)
        self.assertEqual(ad_cfg.AD_microbial_novel_alpha_div_country_ids, [1])


class TestCountryExposureShiftConfig(unittest.TestCase):
    def test_shifted_datasets_differ_from_country_dataset_only_in_tables(self):
        base = ds_cfg.config_novel_alpha_faith_pd_w_geo
        for token in ["m05", "p05"]:
            new = getattr(ds_cfg, f"config_novel_alpha_faith_pd_w_geo_shift_{token}")
            self.assertEqual(new["dataset"], f"ft_vat19_anomaly_v20240806_entero_family_shift_{token}.tsv")
            self.assertEqual(new["dataset_name"], f"microbial_novel_alpha_faith_pd_w_geo_shift_{token}")
            for key in set(base) - {"dataset", "dataset_name"}:
                self.assertEqual(new[key], base[key], key)

    def test_ad_configs_score_the_country_model_on_the_shifted_datasets(self):
        for token in ["m05", "p05"]:
            self.assertEqual(
                getattr(ad_cfg, f"AD_microbial_novel_alpha_div_country_shift_{token}"),
                model_cfg.microbial_models_path_novel_alpha_div_country.rstrip("/") + f"_shift_{token}/")
            self.assertEqual(getattr(ad_cfg, f"AD_microbial_novel_alpha_div_country_shift_{token}_ids"), [1])
            dataset = f"microbial_novel_alpha_faith_pd_w_geo_shift_{token}"
            for name, base in [
                (f"param_list_AD_microbial_novel_alpha_div_country_shift_{token}",
                 ad_cfg.param_list_AD_microbial_novel_alpha_div2),
                (f"param_list_AD_microbial_novel_alpha_div_country_reliability_eval_shift_{token}",
                 ad_cfg.param_list_AD_microbial_novel_alpha_div2_reliability_eval),
            ]:
                expected = [dict(p, dataset=dataset) for p in base]
                self.assertEqual(getattr(ad_cfg, name), expected, name)

