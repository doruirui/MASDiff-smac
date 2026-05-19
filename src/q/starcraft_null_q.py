class StarcraftNullQProvider:
    """星际争霸不需要 Q 表，只提供参数量信息以兼容框架"""
    def __init__(self, param_dim: int, **kwargs):
        self.param_dim = param_dim

    def get_q(self, *args, **kwargs):
        return None

    @property
    def num_actions(self):
        return self.param_dim