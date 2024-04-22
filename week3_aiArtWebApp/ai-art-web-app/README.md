# Lesson 12: Image Generation

In this lesson, we will learn about the image generation models and how to use the OpenAI Image Generation APIs.

We are going to learn about the Stable Diffusion process used in AI image generation, particularly using the DALI·E model provided by OpenAI. We are going to learn how to use the OpenAI Image Generation APIs to generate images from text prompts and from other images.

We'll study how the transformers architecture, which we used in the text generation models, can also be used to run image generation models. We're going to learn about noising and denoising processes and how they have been adapted in the last years to build clever image generation algorithms.

## Prerequisites

* Learn how to use a shell/terminal/console/bash on your device
  * Get familiar with basic commands like `cd`, `ls` and `mkdir`
  * Learn how to execute packages, scripts and commands on your device
* Install python tools on your device
  * [Python](https://www.python.org/downloads/)
  * [Pip](https://pip.pypa.io/en/stable/installation/)
* Learn how to use `python` and `pip` commands
  * Docs: [Python](https://docs.python.org/3/)
  * Docs: [Pip](https://pip.pypa.io/en/stable/)
* Learn how to use `venv` to create and manage virtual environments
  * Docs: [Python venv](https://docs.python.org/3/library/venv.html)
* Install `git` CLI on your device
  * [Git](https://git-scm.com/downloads)
* Learn how to use `git` commands to clone repositories
  * Docs: [Git](https://git-scm.com/doc)
* Learn the basics of `python` programming language syntax
  * [Python official tutorial](https://docs.python.org/3.12/tutorial/index.html)
  * [Learn X in Y minutes](https://learnxinyminutes.com/docs/python/)
* Create an account at [OpenAI Platform](https://platform.openai.com/)
  * If you want to run the API Commands on the platform, set up [billing](https://platform.openai.com/account/billing/overview) and add at least **5 USD** credits to your account

## Review of Lesson 11

* Comparing models
* Code generation models
* Image generation models
* Stable Diffusion
* `Diffusion` python package from Hugging Face
* Running text-to-image models

## Overview of Stable Diffusion for Image Generation

Modern AI models for image generation share similarities with GPT Language Models (LLMs), particularly in their underlying architecture and operational principles.

These models often employ a transformer-based generative architecture, which is specially adapted to handle visual data, as opposed to the textual data processed by GPT models.

The core difference lies in the data handling mechanism. While GPT models generate textual sequences, these image generation models handle pixel or feature sequences.

The image generation process is initiated with a text prompt or a partially completed image. This input is then encoded into a latent space representation, similar to how GPT models encode textual input.

The transformer within the model plays a crucial role in the subsequent steps. It iteratively refines the encoded representation, drawing guidance from the patterns it learned during its training phase.

This refinement process continues until the model generates a coherent and contextually relevant image, aligning with the initial text prompt or the partially completed image.

The ability to generate images from textual descriptions or incomplete images showcases the model's understanding of visual data and its context, demonstrating a significant leap in AI capabilities.

The same way that a GPT can infer tokens related to other tokens, an image generation model can infer pixels and features related to other pixels and features.

Starting from a completely unintelligible image, in each step it predicts and adds details, eventually decoding the latent representation into a more coherent detailed image.

This approach allows the model to generate highly detailed and contextually relevant images from textual descriptions or partial images. It leverages the transformer's ability to handle complex dependencies and relationships, just like GPT models do with text.

### How it Works

Stable diffusion is a process used in AI image generation, particularly in models like Diffusion Models and Denoising Diffusion Probabilistic Models (DDPM).

The process involves two main steps: forward diffusion and backward denoising:

* Forward Diffusion: The forward diffusion process is a noising process
  * It starts with an image (which can be a real image or noise) and gradually adds Gaussian noise over a sequence of steps until the image becomes pure noise
  * This process can be thought of as a Markov chain, where each step is only dependent on the previous one
  * Mathematically, this is modeled by a series of conditional Gaussian distributions
  * The idea is to transform the data distribution (the distribution of natural images, in this case) into a tractable distribution (e.g., isotropic Gaussian) through a sequence of small, simple transformations.

* Backward Denoising: After the forward diffusion, the backward denoising process seeks to reverse the noising process
  * It's essentially a learned denoising process that incrementally converts the noise back into an image
    * To achieve this, the model learns to predict the noise that was added at each step of the forward process and then subtracts it to reverse the diffusion
    * This is done using a neural network, which is trained to estimate the reverse of the diffusion process
    * The network is conditioned on various factors, like the current level of noise in the image and possibly an embedding of textual information if the goal is to generate images from text descriptions
  * The backward process is iterative, starting from pure noise and gradually reducing the noise level until it reconstructs an image that corresponds to the initial data distribution
  * The output is a sample that resembles a natural image from the dataset it was trained on, or if text conditioning is used, an image that corresponds to the text description.

Training the model involves optimizing a loss function, which measures the difference between the predicted noise and the actual noise added during the forward diffusion process

* After training, the model can generate new images by starting from a random Gaussian noise image and iteratively denoising it using the learned denoising process

>The best way to understand visual AI generation is with visual examples. The [The Illustrated Stable Diffusion](http://jalammar.github.io/illustrated-stable-diffusion/) article is a great resource to understand this process visually.

## Using OpenAI Image Generation APIs

* OpenAI API endpoints for [Image Generation](https://platform.openai.com/docs/api-reference/images)
* Generating images with [DALL·E](https://platform.openai.com/docs/guides/images) from text prompts
  * DALL·E-2 vs DALL·E-3
  * Quality
  * Size
  * Styles
* Generating images with other images
  * Edit
    * Prompt
    * Mask
  * Variations

### Generating Images with DALL·E Using Python

* Practical exercises
  * Exercise 1: [Generate Images with DALL·E](./project/00-Generate-Images-DALL-E.md)
    * Code and run [Image-Generation.py](./project/Image-Generation.py) to generate an image from a text prompt
  * Exercise 2: [Generate Image Variations with DALL·E](./project/01-Generate-Image-Variations.md)
    * Code and run [Image-Generation-Variations.py](./project/Image-Generation-Variations.py) to generate variations of an image

## Weekend Project

For consolidating the knowledge acquired in this week, the students should complete the following project:

1. Create a github repository for your project
2. Add all members of your group
3. Create a README.md file that describes your project
4. Create a new application from scratch using NextJS
5. Create an assistant in OpenAI that composes descriptions of paintings

   * The assistant should be able to suggest and describe the details of a painting based on a short description from the user

6. Configure the assistant prompt to be efficient at answering strictly painting descriptions with details about its elements, style, details, and colors
7. Create a page for the user to pick an option from a selection of painting themes
8. Include a button to send a message to the assistant for it to generate a painting description with the selected theme

   * Use the OpenAI [Assistants API](https://platform.openai.com/docs/assistants/overview) to create threads and messages for the assistant

9. When the assistant answers, display the generated text output in a text box
10. Create a button for the user to request the image to be generated based on the content of the text box
11. Create a short form after the button for the user to choose the image generation parameters
12. Include a button to request the image to be generated
13. Ask the Image Generation API to generate the image
14. Display a loader while the image is being generated
15. Display the image in the page when it is generated
16. Submit your project in the submission form

>You should find your group in the [Discord](https://discord.gg/encodeclub) AI Bootcamp Channel
>>If you can't find your group, please contact the program manager through Discord or email

## Next Steps

* Generating images with AI
* Running models locally
* Using Stable Diffusion WebUI
* Image generation models
* Prompting techniques for image generation

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Next, open [http://localhost:3000](http://localhost:3000) with your browser to see the result.