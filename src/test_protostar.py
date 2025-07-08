import pytest
from line_profiler import profile
import protostar
import tools
import os

TEST_OBJ_DIRECTORY = '/Users/namsonnguyen/repo/OutflowPolyfit/dataSpring2025/'

@pytest.fixture
def protostar_object():
    star = 'HOPS-50'
    return tools.open_pickle(f'{TEST_OBJ_DIRECTORY}/{star}')

@profile
def test_name(protostar_object):
    expected_name = 'HOPS-50'
    actual_name = protostar_object.name
    assert expected_name == actual_name