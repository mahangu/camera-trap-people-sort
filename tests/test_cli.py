"""Tests for the camera-trap-people-sort CLI."""

import os
from pathlib import Path
import pytest
from camera_trap_people_sort.cli import parse_args

def test_parse_args():
    """Test argument parsing."""
    args = parse_args(['--input-path', 'tests/input', '--output-path', 'tests/output'])
    assert args.input_path == 'tests/input'
    assert args.output_path == 'tests/output'
    assert not args.recursive

def test_parse_args_recursive():
    """Test argument parsing with recursive flag."""
    args = parse_args(['--input-path', 'tests/input', '--recursive'])
    assert args.recursive
    assert args.use_yolo  # Default model should be enabled

def test_parse_args_model_selection():
    """Test model selection arguments."""
    args = parse_args(['--yolo-only'])
    assert args.use_yolo
    assert not args.use_pose
    assert not args.use_megadetector
    assert not args.use_frcnn

def test_parse_args_all_models_disabled():
    """Test that YOLOv8x is used as fallback when all models are disabled."""
    args = parse_args(['--no-yolo', '--no-pose', '--no-megadetector', '--no-frcnn'])
    assert args.use_yolo  # YOLO should be enabled as fallback
    assert not args.use_pose
    assert not args.use_megadetector
    assert not args.use_frcnn

def test_parse_args_confidence_thresholds():
    """Test confidence threshold arguments."""
    args = parse_args(['--confidence', '0.5', '--min-confidence', '0.3'])
    assert args.confidence == 0.5
    assert args.min_confidence == 0.3
