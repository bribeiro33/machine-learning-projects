
import unittest
from pathlib import Path
import sys
sys.path.append('..')
from Grid_response import *

class TestGrid2(unittest.TestCase):
    def setUp(self):
        self.grid = Grid("./test_files/grid_2.test")


    def test_2_bfs(self):
        output = self.grid.bfs()
        output_str = str(output)
        correct_str = Path("./test_files/raw/2_bfs.ans").read_text()
        correct_str = correct_str.replace('\n', '')
        self.assertEqual(output_str, correct_str)


    def test_2_dfs(self):
        output = self.grid.dfs()
        output_str = str(output)
        correct_str = Path("./test_files/raw/2_dfs.ans").read_text()
        correct_str = correct_str.replace('\n', '')
        self.assertEqual(output_str, correct_str)


    def test_2_ucs(self):
        output = self.grid.ucs()
        output_str = str(output)
        correct_str = Path("./test_files/raw/2_ucs.ans").read_text()
        correct_str = correct_str.replace('\n', '')
        self.assertEqual(output_str, correct_str)


    def test_2_astar(self):
        output = self.grid.astar()
        output_str = str(output)
        correct_str = Path("./test_files/raw/2_astar.ans").read_text()
        correct_str = correct_str.replace('\n', '')
        self.assertEqual(output_str, correct_str)