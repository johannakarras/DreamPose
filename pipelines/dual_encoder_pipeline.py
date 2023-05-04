import inspect
from typing import Callable, List, Optional, Union
from einops import rearrange

import numpy as np
import torch, torchvision
import torch.nn.functional as F
from  torch.cuda.amp import autocast
from torchvision import transforms
from torchvision.utils import make_grid

import PIL
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPProcessor

from diffusers.configuration_utils import FrozenDict
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DiffusionPipeline
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

from diffusers.utils import PIL_INTERPOLATION, deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from models.unet_dual_encoder import get_unet, Embedding_Adapter

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class StableDiffusionImg2ImgPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-guided image to image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker"]

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.__init__
    def __init__(
        self,
        #adapter: Embedding_Adapter,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModel,
        clip_processor: CLIPProcessor,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        safety_checker: None,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = False,
        stochastic_sampling: bool = False
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.adapter = Embedding_Adapter().cuda()

        self.register_modules(
            #adapter=self.adapter,
            vae=vae,
            image_encoder=image_encoder,
            clip_processor=clip_processor,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        self.vae = self.vae.cuda()
        self.unet = self.unet.cuda()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        self.fixed_noise = None
        self.stochastic_sampling = stochastic_sampling

        print("Stochastic Sampling: ", self.stochastic_sampling)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_sequential_cpu_offload
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.image_encoder, self.clip_processor, self.vae, self.adapter]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            # TODO(Patrick) - there is currently a bug with cpu offload of nn.Parameter in accelerate
            # fix by only offloading self.safety_checker for now
            cpu_offload(self.safety_checker.vision_model, device)

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_image(self, image, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            image (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(image) if isinstance(image, list) else 1
        #print("Batch size = ", batch_size)

        if isinstance(image, list):
            uncond_image = [torch.zeros((image[0].size[0], image[0].size[1], 3)) for _ in range(batch_size)]
        else:
            image = [image]
            uncond_image = [torch.zeros((image[0].size[0], image[0].size[1], 3))]

        with autocast():
            # clip encoder
            inputs = self.processor(images=image, return_tensors="pt")
            clip_image_embeddings = self.clip_encoder(**inputs).last_hidden_state.cuda()

            uncond_inputs = self.processor(images=uncond_image, return_tensors="pt")
            clip_uncond_image_embeddings = self.clip_encoder(**uncond_inputs).last_hidden_state.cuda()

            # vae encoder
            #image = torch.from_numpy(np.array(image).transpose((2,0,1))) 
            #image = image.cuda().float().unsqueeze(0)
            image_tensor = torch.tensor([np.array(im).transpose((2,0,1)) for im in image]).cuda().float()
            vae_image_embeddings = self.vae.encode(image_tensor).latent_dist.sample() * 0.18215
            #vae_image_embeddings = rearrange(image_embeddings, 'b h w c -> b (h w) c')

            #img_shape = image.shape
            #uncond_image = torch.zeros(img_shape).cuda().float()
            #uncond_image = uncond_image.cuda().float()
            uncond_image_tensor = torch.tensor([np.array(im).transpose((2,0,1)) for im in uncond_image]).cuda().float()
            vae_uncond_image_embeddings = self.vae.encode(uncond_image_tensor).latent_dist.sample() * 0.18215
            #vae_uncond_image_embeddings = rearrange(uncond_image_embeddings, 'b h w c -> b (h w) c')

            # adapt embeddings
            image_embeddings = self.adapter(clip_image_embeddings, vae_image_embeddings)
            uncond_image_embeddings = self.adapter(clip_uncond_image_embeddings, vae_uncond_image_embeddings)

        #print(image_embeddings.shape)
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings  = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        bs_embed, seq_len, _ = uncond_image_embeddings .shape
        uncond_image_embeddings  = uncond_image_embeddings.repeat(1, num_images_per_prompt, 1)
        uncond_image_embeddings = uncond_image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            image_embeddings = torch.cat([uncond_image_embeddings, image_embeddings, image_embeddings])

        return image_embeddings

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        with autocast():
            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, strength, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [1.0, 1.0] but is {strength}")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        with autocast():
            image = image.to(device=device, dtype=dtype).cuda()
            init_latent_dist = self.vae.encode(image).latent_dist
            init_latents = init_latent_dist.sample(generator=generator)
            init_latents = 0.18215 * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt * num_images_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)

        # add noise to latents using the timesteps
        if self.fixed_noise is None:
            #print("Latents Shape = ", init_latents.shape, init_latents[0].shape, image.shape[0])
            single_fixed_noise = torch.randn(init_latents[0].shape, generator=generator, device=device, dtype=dtype)
            self.fixed_noise = single_fixed_noise.repeat(image.shape[0], 1, 1, 1)#torch.tensor([single_fixed_noise for _ in range(image.shape[0])])
        noise = self.fixed_noise

        # get latents
        init_latents = self.scheduler.add_noise(init_latents.cuda(), noise.cuda(), timestep)
        latents = init_latents

        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        adapter = None,
        prev_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        pose: Union[torch.FloatTensor, PIL.Image.Image] = None,
        strength: float = 1.0,
        num_inference_steps: Optional[int] = 100,
        guidance_scale: Optional[float] = 7.5,
        s1: float = 1.0, # strength of input pose
        s2: float = 1.0, # strength of input image
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        frames = [],
        sweep = False,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # 1. Check inputs
        self.check_inputs(prompt, strength, callback_steps)

        # 2. Set adapter
        if adapter is not None:
            print("Setting adapter")
            self.adapter = adapter

        # 3. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0 or s1 > 0.0 or s2 > 0.0

        # 4. Encode input image: [unconditional, condional, conditional]
        embeddings = self._encode_image(
            image, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 5. Preprocess image
        image = preprocess(image)
        pose = preprocess(pose)
        image, pose = torch.tensor(image), torch.tensor(pose)

        # 6. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 7. Prepare latent variables
        latents = self.prepare_latents(
            image, latent_timestep, batch_size, num_images_per_prompt, embeddings.dtype, device, generator
        )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. If sweeping (s1, s2) values, prepare variables
        if sweep:
            s1_vals = [0, 3, 5, 7, 9] 
            s2_vals = [0, 3, 5, 7, 9]
            images = [] # store frames
        else:
            s1_vals, s2_vals = [s1], [s2]

        # 10. Denoising loop
        copy_latents = latents.clone()
        for s1 in s1_vals:
            for s2 in s2_vals:
                latents = copy_latents.clone()
                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        t = t.cuda()

                        # If stochastic sampling enabled, randomly select from previous images
                        if self.stochastic_sampling:
                            idx = np.random.choice(range(len(frames)))
                            input_image = [frames[idx] for _ in range(len(image))]
                            #print("Selecting conditioning image #", idx)
                            embeddings = self._encode_image(
                                input_image, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
                            )
                            input_image = preprocess(input_image)

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        # Add pose to noisy latents
                        _, _, h, w = latent_model_input.shape
                        if do_classifier_free_guidance:
                            pose_input = torch.cat([torch.zeros(pose.shape), pose, torch.zeros(pose.shape)]) 
                        else:
                            pose_input = torch.cat([pose, pose, pose]) 
                        latent_model_input = torch.cat((latent_model_input.cuda(), F.interpolate(pose_input, (h,w)).cuda()), 1)

                        # predict the noise residual
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=embeddings.cuda()).sample

                        # perform guidance
                        if do_classifier_free_guidance:
                            #print(f"s1={s1}, s2={s2}")
                            noise_pred_uncond, noise_pred_, noise_pred_img_only = noise_pred.chunk(3)
                            noise_pred = noise_pred_uncond + \
                                         s1 * (noise_pred_img_only - noise_pred_uncond) + \
                                         s2 * (noise_pred_ - noise_pred_img_only) 

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred.cuda(), t, latents.cuda(), **extra_step_kwargs).prev_sample

                        # call the callback, if provided
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and i % callback_steps == 0:
                                callback(i, t, latents)

                # 11. Post-processing
                latents = latents[:,:4, :, :].cuda() #.float()
                image = self.decode_latents(latents)

                #print(len(image)) # 1
                #print(image[0].shape) # 640, 512, 3

                # 13. Convert to PIL
                if output_type == "pil":
                    image = self.numpy_to_pil(image)

                if sweep:
                    images.append(torchvision.transforms.ToTensor()(image[0]).clone())

        # 13. If sweeping, convert images to grid
        if sweep:
            Grid = make_grid(images, nrow=len(s2_vals))
            image = [torchvision.transforms.ToPILImage()(Grid)]
            #image = Grid
            #print("Grid complete.")

        # 14. Run safety checker
        #image, has_nsfw_concept = self.run_safety_checker(image, device, embeddings.dtype)

        if not return_dict:
            return (image, False)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=False)

