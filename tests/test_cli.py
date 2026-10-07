"""Tests for ``cli``: the option helpers the three commands share."""

import argparse

import pytest

from netcdf2region.cli import add_option


def _parser():
    parser = argparse.ArgumentParser()
    add_option(parser, "--n-jobs", legacy="--n_jobs", type=int, default=8, help="jobs")
    return parser


def test_both_spellings_set_the_same_value():
    parser = _parser()

    assert parser.parse_args([]).n_jobs == 8
    assert parser.parse_args(["--n-jobs", "2"]).n_jobs == 2
    assert parser.parse_args(["--n_jobs", "3"]).n_jobs == 3


def test_the_legacy_spelling_is_hidden_from_help():
    help_text = _parser().format_help()

    assert "--n-jobs" in help_text
    assert "--n_jobs" not in help_text


def test_a_required_option_cannot_have_a_legacy_alias():
    with pytest.raises(ValueError, match="required"):
        add_option(argparse.ArgumentParser(), "--out-file", legacy="--out_file", required=True)
