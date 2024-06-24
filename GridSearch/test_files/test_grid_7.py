
import unittest
from pathlib import Path
import sys
sys.path.append('..')
from Grid_response import *

class TestGrid7(unittest.TestCase):
    def setUp(self):
        self.grid = Grid("./test_files/grid_7.test")


    def test_7_bfs(self):
        output = self.grid.bfs()
        output_str = str(output)
        correct_str = Path("./test_files/raw/7_bfs.ans").read_text()
        correct_str = correct_str.replace('\n', '')
        self.assertEqual(output_str, correct_str)


    def test_7_dfs(self):
        output = self.grid.dfs()
        output_str = str(output)
        correct_str = Path("./test_files/raw/7_dfs.ans").read_text()
        correct_str = correct_str.replace('\n', '')
        self.assertEqual(output_str, correct_str)


    def test_7_ucs(self):
        output = self.grid.ucs()
        output_str = str(output)
        correct_str = Path("./test_files/raw/7_ucs.ans").read_text()
        correct_str = correct_str.replace('\n', '')
        self.assertEqual(output_str, correct_str)


    def test_7_astar(self):
        output = self.grid.astar()
        output_str = str(output)
        correct_str = Path("./test_files/raw/7_astar.ans").read_text()
        correct_str = correct_str.replace('\n', '')
        self.assertEqual(output_str, correct_str)