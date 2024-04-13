# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from diffusion.respace import *
import torch
from diffusion.gaussian_diffusion import ModelVarType, ModelMeanType


class GaussianDiffusionSS(SpacedDiffusion):
    """
        Utilities for training and sampling diffusion models.

        Ported directly from here, and then adapted over time to further experimentation.
        https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

        :param betas: a 1-D numpy array of betas for each diffusion timestep,
                      starting at T and going to 1.
        :param model_mean_type: a ModelMeanType determining what the model outputs.
        :param model_var_type: a ModelVarType determining how variance is output.
        :param loss_type: a LossType determining the loss function to use.
        :param rescale_timesteps: if True, pass floating point timesteps into the
                                  model so that they are always scaled like in the
                                  original paper (0 to 1000).
        """

    def __init__(self,
                 use_timesteps,
                 *,
                 betas,
                 model_mean_type,
                 model_var_type,
                 loss_type,
                 ):
        super(GaussianDiffusionSS, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                            model_mean_type=model_mean_type,
                                                            model_var_type=model_var_type,
                                                            loss_type=loss_type, )

    def p_sample(
        self,
        model,
        x,
        t,
        gamma_factor=0.0,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            pred_xstart = out['pred_xstart']
            variance = out['variance'][0][0][0][0].item()
            self.max_variance = max(self.max_variance, variance)
            # adding sin timely-decay factor to the guidance schedule
            current_time = t[0].item()
            add_value = max(np.sin((current_time/self.num_timesteps)*np.pi)*self.max_variance*gamma_factor, 0.0)
            # off-the-shelf classifier guidance
            gradient = cond_fn([x,pred_xstart], self._scale_timesteps(t), **model_kwargs)
            out["mean"] = (
                out["mean"].float() + (out["variance"]+add_value)*gradient.float()
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        self.max_variance = 0.0
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]


class GaussianDiffusionSSNorm(GaussianDiffusionSS):
    """
        Utilities for training and sampling diffusion models.

        Ported directly from here, and then adapted over time to further experimentation.
        https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

        :param betas: a 1-D numpy array of betas for each diffusion timestep,
                      starting at T and going to 1.
        :param model_mean_type: a ModelMeanType determining what the model outputs.
        :param model_var_type: a ModelVarType determining how variance is output.
        :param loss_type: a LossType determining the loss function to use.
        :param rescale_timesteps: if True, pass floating point timesteps into the
                                  model so that they are always scaled like in the
                                  original paper (0 to 1000).
        """

    def __init__(self,
                 use_timesteps,
                 *,
                 betas,
                 model_mean_type,
                 model_var_type,
                 loss_type,
                 ):
        super(GaussianDiffusionSSNorm, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                            model_mean_type=model_mean_type,
                                                            model_var_type=model_var_type,
                                                            loss_type=loss_type, )

    def process_xstart(self, x, denoised_fn=None, clip_denoised=None):
        if denoised_fn is not None:
            x = denoised_fn(x)
        if clip_denoised:
            return x.clamp(-1, 1)
        return x

    def p_sample(
        self,
        model,
        x,
        t,
        gamma_factor=0.0,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            pred_xstart = out['pred_xstart']
            variance = out['variance'][0][0][0][0].item()
            self.max_variance = max(self.max_variance, variance)
            # adding sin timely-decay factor to the guidance schedule
            current_time = t[0].item()
            add_value = max(np.sin((current_time/self.num_timesteps)*np.pi)*self.max_variance*gamma_factor, 0.0)

            eps_gen =self._predict_eps_from_xstart(x, t, out['pred_xstart'])
            delta_eps = out["delta_e"]
            cfg_s = out["cfgs"]


            # off-the-shelf classifier guidance
            gradient = cond_fn([x,pred_xstart], self._scale_timesteps(t), **model_kwargs)
            norm_delta_eps = delta_eps.pow(2).sum().sqrt()
            norm_gradient = gradient.pow(2).sum().sqrt()
            gradient = gradient * (norm_delta_eps/norm_gradient) * (out["variance"] + add_value)

            new_eps = eps_gen + cfg_s * gradient
            new_predxstart = self.process_xstart(self._predict_xstart_from_eps(x, t, new_eps), denoised_fn, clip_denoised)
            out["mean"], _, _ = self.q_posterior_mean_variance(x_start=new_predxstart, x_t=x, t=t)
            out["pred_xstart"] = new_predxstart
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_mean_variance_ss_normalize(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
                Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
                the initial x, x_0.
                :param model: the model, which takes a signal and a batch of timesteps
                              as input.
                :param x: the [N x C x ...] tensor at time t.
                :param t: a 1-D Tensor of timesteps.
                :param clip_denoised: if True, clip the denoised signal into [-1, 1].
                :param denoised_fn: if not None, a function which applies to the
                    x_start prediction before it is used to sample. Applies before
                    clip_denoised.
                :param model_kwargs: if not None, a dict of extra keyword arguments to
                    pass to the model. This can be used for conditioning.
                :return: a dict with the following keys:
                         - 'mean': the model mean output.
                         - 'variance': the model variance output.
                         - 'log_variance': the log of 'variance'.
                         - 'pred_xstart': the prediction for x_0.
                """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output, delta_e, cfg_scale = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
            "delta_e": delta_e,
            "cfgs": cfg_scale
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return self.p_mean_variance_ss_normalize(self._wrap_model(model), *args, **kwargs)

class GaussianDiffusionSSNormv2(GaussianDiffusionSSNorm):
    """
        Utilities for training and sampling diffusion models.

        Ported directly from here, and then adapted over time to further experimentation.
        https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

        :param betas: a 1-D numpy array of betas for each diffusion timestep,
                      starting at T and going to 1.
        :param model_mean_type: a ModelMeanType determining what the model outputs.
        :param model_var_type: a ModelVarType determining how variance is output.
        :param loss_type: a LossType determining the loss function to use.
        :param rescale_timesteps: if True, pass floating point timesteps into the
                                  model so that they are always scaled like in the
                                  original paper (0 to 1000).
        """

    def __init__(self,
                 use_timesteps,
                 *,
                 betas,
                 model_mean_type,
                 model_var_type,
                 loss_type,
                 ):
        super(GaussianDiffusionSSNormv2, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                            model_mean_type=model_mean_type,
                                                            model_var_type=model_var_type,
                                                            loss_type=loss_type, )

    def p_sample(
            self,
            model,
            x,
            t,
            gamma_factor=0.0,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            pred_xstart = out['pred_xstart']
            variance = out['variance'][0][0][0][0].item()
            self.max_variance = max(self.max_variance, variance)
            # adding sin timely-decay factor to the guidance schedule
            current_time = t[0].item()


            eps_gen = self._predict_eps_from_xstart(x, t, out['pred_xstart'])
            delta_eps = out["delta_e"]
            cfg_s = out["cfgs"]

            # off-the-shelf classifier guidance
            gradient = cond_fn([x, pred_xstart], self._scale_timesteps(t), **model_kwargs)
            norm_delta_eps = delta_eps.pow(2).sum().sqrt()
            norm_gradient = gradient.pow(2).sum().sqrt()
            gradient = gradient * (norm_delta_eps / norm_gradient)

            new_eps = eps_gen + cfg_s * gradient
            new_predxstart = self.process_xstart(self._predict_xstart_from_eps(x, t, new_eps), denoised_fn,
                                                 clip_denoised)
            out["mean"], _, _ = self.q_posterior_mean_variance(x_start=new_predxstart, x_t=x, t=t)
            out["pred_xstart"] = new_predxstart
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)
