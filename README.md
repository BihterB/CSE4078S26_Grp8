# CSE4078S26_Grp8

CSE4078 Spring 2026 Term Project

## Project Topic

Evaluation and fine-tuning of small open-source Large Language Models for Turkish legal question answering.

## Group

Group 8

## Mandatory Dataset

Since Group 8 is an even-numbered group, we use:

- Renicames/turkish-law-chatbot

## Selected Baseline Models

- Qwen/Qwen2.5-1.5B-Instruct
- google/gemma-2-2b-it

## Deadline 1 Goal

For the progress presentation, we evaluate both base models on the mandatory Turkish legal test corpus.
Then, we compare baseline results and select one model for fine-tuning.

## Planned Fine-tuning Model

Based on baseline results, we plan to fine-tune one selected model using the training split of the mandatory SFT corpus.

## Important Rule

The test split is only used for evaluation.
It is not used for training, fine-tuning, prompt tuning, or adaptation.