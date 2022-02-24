from logging import getLogger
import os
import hashlib
import json

import lightgbm as lgb
import optuna

logger = getLogger(__name__)

DEFAULT_FEATURE_FRACTION = 0.8
DEFAULT_NUM_LEAVES = 2 ** 3
DEFAULT_BAGGING_FRACTION = 1.
DEFAULT_BAGGING_FREQ = 1
DEFAULT_LAMBDA_L1 = 0.
DEFAULT_LAMBDA_L2 = 0.
DEFAULT_MIN_CHILD_SAMPLES = 10
DEFAULT_DIRECTION = 'maximize'


class LGBMTuner:
	def __init__(self, data_path: str, trial_prefix: str, trial_date_iso: str, init_params: dict, screener_hashkey: str):
		"""
		:param data_path: データ保存先
		:param trial_prefix: 試行ごとにつける任意の文字列
		:param trial_date_iso: 試行日
		:param screener_hashkey: 前処理プロセスで生成されるハッシュキー。以前の処理を識別するための任意の文字列でよい
		"""
		self.data_path = data_path
		os.makedirs(self.data_path, exist_ok=True)

		self.trial_prefix = trial_prefix
		self.trial_date_iso = trial_date_iso
		self.init_params = init_params
		self.screener_hashkey = screener_hashkey

		logger.debug('data_path: {}'.format(self.data_path))
		logger.debug('trial_prefix: {}'.format(self.trial_prefix))
		logger.debug('trial_date_iso: {}'.format(self.trial_date_iso))
		logger.debug('screener_hashkey: {}'.format(self.screener_hashkey))
		logger.debug('init_params: {}'.format(self.init_params))

		self.hyper_params = dict()
		self.direction = DEFAULT_DIRECTION
		self.n_trials = 0

		self.best_params = {
			'feature_fraction': DEFAULT_FEATURE_FRACTION,
			'num_leaves': DEFAULT_NUM_LEAVES,
			'bagging_fraction': DEFAULT_BAGGING_FRACTION,
			'bagging_freq': DEFAULT_BAGGING_FREQ,
			'lambda_l1': DEFAULT_LAMBDA_L1,
			'lambda_l2': DEFAULT_LAMBDA_L2,
			'min_child_samples': DEFAULT_MIN_CHILD_SAMPLES,
		}
		self.best_params.update(init_params)

		logger.debug('default_params: {}'.format(self.best_params))

	def get_hashkey(self):
		hash_seed = json.dumps(self.hyper_params_list) + json.dumps(self.init_params) + self.direction + str(self.n_trials) + self.screener_hashkey
		hashkey = hashlib.md5(hash_seed.encode('utf-8')).hexdigest()
		return hashkey

	def set_data(self, train_set, valid_sets, valid_names):
		"""
		:param train_set: 学習用データ
		:param valid_sets: 検証用データ
		:param valid_names: 検証用データの名前
		"""
		self.train_set = train_set
		self.valid_sets = valid_sets
		self.valid_names = valid_names

		logger.debug('train_set: {}'.format(train_set))
		logger.debug('valid_sets: {}'.format(valid_sets))
		logger.debug('valid_names: {}'.format(valid_names))

	def evaluate(self, model, params):
		score = 0
		# ToDo: ndgc以外のスコアを計算する
		if params['metric'] == 'ndcg':
			for eval_at in params['ndcg_eval_at']:
				score += model.best_score['valid']['ndcg@{}'.format(eval_at)]
			logger.debug('score: {}'.format(score))
		else:
			raise ValueError('metric: {} is not supported'.format(params['metric']))

		return score

	def objective(self, trial):
		params = self.best_params.copy()
		for param in self.current_hyper_params:
			if param == 'feature_fraction':
				params.update({param: trial.suggest_uniform(param, 0.4, 1.)})
			elif param == 'num_leaves':
				params.update({param: trial.suggest_int(param, 2, 2 ** 8)})
			elif param == 'bagging_fraction':
				params.update({param: trial.suggest_uniform(param, 0.4, 1.)})
			elif param == 'bagging_freq':
				params.update({param: trial.suggest_int(param, 1, 7)})
			elif param == 'lambda_l1':
				params.update({param: trial.suggest_uniform(param, 0., 10.)})
			elif param == 'lambda_l2':
				params.update({param: trial.suggest_uniform(param, 0., 10.)})
			elif param == 'min_child_samples':
				params.update({param: trial.suggest_int(param, 5, 100)})
		model = lgb.train(
			params=params,
			train_set=self.train_set,
			valid_sets=self.valid_sets,
			valid_names=self.valid_names,
			callbacks=self.callbacks,
		)

		score = self.evaluate(model, params)
		return score

	def train(self, callbacks, hyper_params_list, n_total_trials, direction=DEFAULT_DIRECTION, refresh_study=False, overwrite_model=False):
		"""
		:param callbacks: コールバック
		:param hyper_params_list: ハイパーパラメータのリスト
		:param n_trials: 試行回数
		:param direction: minimize or maximize
		:param refresh_study: studyを再作成するかどうか
		:param overwrite_model: 学習済みモデルを上書きするかどうか
		:return:
		"""
		logger.debug('start params: {}'.format(self.best_params))
		logger.debug('callbacks: {}'.format(callbacks))
		logger.debug('hyper_params_list: {}'.format(hyper_params_list))
		logger.debug('n_total_trials: {}'.format(n_total_trials))
		logger.debug('direction: {}'.format(direction))

		self.callbacks = callbacks
		self.hyper_params_list = hyper_params_list
		self.n_trials = int(n_total_trials / len(hyper_params_list))
		self.direction = direction

		hashkey = self.get_hashkey()
		study_name = '{}_{}_{}'.format(self.trial_prefix, self.trial_date_iso, hashkey)

		study_storage_file = os.path.join(self.data_path, 'optuna_study_{}.db'.format(study_name))
		if os.path.exists(study_storage_file) and refresh_study:
			os.remove(study_storage_file)
			logger.debug('remove study_storage_file: {}'.format(study_storage_file))

		study = optuna.create_study(
			study_name=study_name,
			storage='sqlite:///{}'.format(study_storage_file),
			load_if_exists=True,
			direction=direction,
		)

		self.current_hyper_params = list()		# チューニング対象のハイパーパラメータ
		for hyper_params in hyper_params_list:
			self.current_hyper_params = hyper_params
			logger.debug('current_hyper_params: {}'.format(self.current_hyper_params))

			study.optimize(self.objective, n_trials=self.n_trials, timeout=None)
			logger.debug('study.best_params: {}'.format(study.best_params))
			self.best_params.update(study.best_params.copy())

		self.study = study
		lgb_model = lgb.train(self.best_params, self.train_set, valid_sets=self.valid_sets, valid_names=self.valid_names, callbacks=callbacks)

		# モデルの保存先
		model_file = os.path.join(self.data_path, 'model_{}.txt'.format(study_name))
		if os.path.exists(model_file) and overwrite_model is False:
			logger.debug('model_file: {} exists'.format(model_file))
		else:
			lgb_model.save_model(model_file)
			logger.debug('model_file: {} saved'.format(model_file))
			lgb_model = lgb.Booster(model_file=model_file)

		return lgb_model


def test():
	logger.debug("test")


if __name__ == '__main__':
	test()
