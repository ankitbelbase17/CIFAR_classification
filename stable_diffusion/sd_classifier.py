"""
stable_diffusion/sd_classifier.py - Stable Diffusion for classification (inference-only)
Uses CLIP embeddings and diffusion model features for zero-shot classification
"""

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class StableDiffusionClassifier:
    """
    Zero-shot classifier using Stable Diffusion (inference-only, no training)
    
    Uses two approaches:
    1. CLIP text-image similarity
    2. Diffusion reconstruction error per class
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1-base",
        device: str = "cuda"
    ):
        """
        Initialize Stable Diffusion classifier
        
        Args:
            model_id: Hugging Face model ID for Stable Diffusion
            device: Device to run on
        """
        self.device = device
        print(f"Loading Stable Diffusion model: {model_id}")
        
        # Load Stable Diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.pipe = self.pipe.to(device)
        
        # Use DDIM for faster inference
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Extract CLIP model
        self.clip_model = self.pipe.text_encoder
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # For reconstruction-based classification
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        
        print("✓ Stable Diffusion classifier initialized")
    
    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get CLIP text embeddings for class names"""
        inputs = self.pipe.tokenizer(
            texts,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.clip_model(**inputs)[0]
        
        return text_embeddings
    
    def get_image_embedding(self, image: Image.Image) -> torch.Tensor:
        """Get CLIP image embedding"""
        # Process image
        inputs = self.clip_processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            # Use VAE encoder to get image features
            if image.size != (512, 512):
                image = image.resize((512, 512), Image.LANCZOS)
            
            # Convert to tensor
            image_tensor = self.clip_processor.image_processor(
                images=image,
                return_tensors="pt"
            )['pixel_values'].to(self.device)
            
            # Encode with VAE
            latents = self.vae.encode(image_tensor.half() if self.device == "cuda" else image_tensor)
            latents = latents.latent_dist.sample() * self.vae.config.scaling_factor
        
        return latents
    
    def clip_similarity_classification(
        self,
        image: Image.Image,
        class_names: List[str],
        templates: List[str] = None
    ) -> Dict:
        """
        Classify using CLIP text-image similarity
        
        Args:
            image: Input image
            class_names: List of class names
            templates: Text templates for class names
        
        Returns:
            Dictionary with predictions and scores
        """
        if templates is None:
            templates = [
                "a photo of a {}",
                "a photograph of a {}",
                "an image of a {}",
                "{}"
            ]
        
        # Generate text prompts for each class
        all_texts = []
        for class_name in class_names:
            for template in templates:
                all_texts.append(template.format(class_name))
        
        # Get text embeddings
        text_embeddings = self.get_text_embeddings(all_texts)
        
        # Process image with CLIP
        inputs = self.clip_processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            # Get image features from CLIP
            image_features = self.pipe.image_encoder(
                inputs['pixel_values']
            ).image_embeds if hasattr(self.pipe, 'image_encoder') else None
            
            if image_features is None:
                # Use VAE features as fallback
                image_features = self.get_image_embedding(image)
                image_features = image_features.flatten(1).mean(1, keepdim=True)
        
        # Calculate similarities for each class (average over templates)
        similarities = []
        for i in range(len(class_names)):
            class_text_embeds = text_embeddings[i * len(templates):(i + 1) * len(templates)]
            
            # Average template embeddings
            class_text_embed = class_text_embeds.mean(0, keepdim=True)
            
            # Compute similarity
            sim = F.cosine_similarity(
                image_features.flatten(1),
                class_text_embed.flatten(1)
            )
            similarities.append(sim.item())
        
        similarities = torch.tensor(similarities)
        probs = F.softmax(similarities * 100, dim=0)  # Temperature scaling
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probs, k=min(5, len(class_names)))
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class': class_names[idx.item()],
                'probability': prob.item(),
                'similarity': similarities[idx.item()].item()
            })
        
        return {
            'method': 'clip_similarity',
            'predictions': predictions,
            'all_probabilities': probs.tolist()
        }
    
    def reconstruction_classification(
        self,
        image: Image.Image,
        class_names: List[str],
        num_steps: int = 20
    ) -> Dict:
        """
        Classify based on reconstruction error with class-conditional prompts
        
        The class that produces the lowest reconstruction error is predicted
        
        Args:
            image: Input image
            class_names: List of class names
            num_steps: Number of denoising steps
        
        Returns:
            Dictionary with predictions
        """
        # Encode image to latents
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.LANCZOS)
        
        image_tensor = self.clip_processor.image_processor(
            images=image,
            return_tensors="pt"
        )['pixel_values'].to(self.device)
        
        with torch.no_grad():
            # Encode
            latents = self.vae.encode(
                image_tensor.half() if self.device == "cuda" else image_tensor
            ).latent_dist.sample() * self.vae.config.scaling_factor
            
            reconstruction_errors = []
            
            # For each class, compute reconstruction
            for class_name in class_names:
                prompt = f"a photo of a {class_name}"
                
                # Get text embeddings
                text_inputs = self.pipe.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                text_embeddings = self.clip_model(text_inputs.input_ids)[0]
                
                # Add noise and denoise (partial reconstruction)
                noise = torch.randn_like(latents)
                timestep = torch.tensor([num_steps]).to(self.device)
                
                noisy_latents = self.pipe.scheduler.add_noise(
                    latents, noise, timestep
                )
                
                # Denoise
                for t in range(num_steps):
                    noise_pred = self.unet(
                        noisy_latents,
                        timestep,
                        encoder_hidden_states=text_embeddings
                    ).sample
                    
                    # Update latents (simplified)
                    noisy_latents = noisy_latents - noise_pred * 0.1
                
                # Compute reconstruction error
                error = F.mse_loss(noisy_latents, latents)
                reconstruction_errors.append(error.item())
            
            # Lower error = better match
            errors_tensor = torch.tensor(reconstruction_errors)
            # Invert errors to get scores (higher = better)
            scores = -errors_tensor
            probs = F.softmax(scores * 10, dim=0)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs, k=min(5, len(class_names)))
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append({
                    'class': class_names[idx.item()],
                    'probability': prob.item(),
                    'reconstruction_error': reconstruction_errors[idx.item()]
                })
        
        return {
            'method': 'reconstruction',
            'predictions': predictions,
            'all_probabilities': probs.tolist()
        }
    
    def classify(
        self,
        image: Image.Image,
        class_names: List[str],
        method: str = "clip",
        **kwargs
    ) -> Dict:
        """
        Main classification method
        
        Args:
            image: Input image
            class_names: List of class names
            method: 'clip' or 'reconstruction' or 'ensemble'
            **kwargs: Additional arguments for specific methods
        
        Returns:
            Classification results
        """
        if method == "clip":
            return self.clip_similarity_classification(image, class_names, **kwargs)
        
        elif method == "reconstruction":
            return self.reconstruction_classification(image, class_names, **kwargs)
        
        elif method == "ensemble":
            # Combine both methods
            clip_results = self.clip_similarity_classification(image, class_names)
            recon_results = self.reconstruction_classification(image, class_names)
            
            # Average probabilities
            clip_probs = torch.tensor(clip_results['all_probabilities'])
            recon_probs = torch.tensor(recon_results['all_probabilities'])
            
            ensemble_probs = (clip_probs + recon_probs) / 2
            top_probs, top_indices = torch.topk(ensemble_probs, k=min(5, len(class_names)))
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append({
                    'class': class_names[idx.item()],
                    'probability': prob.item()
                })
            
            return {
                'method': 'ensemble',
                'predictions': predictions,
                'clip_results': clip_results,
                'reconstruction_results': recon_results
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")

