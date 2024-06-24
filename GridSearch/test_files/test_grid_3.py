
import unittest
from pathlib import Path
import sys
sys.path.append('..')
from Grid_response import *

class TestGrid3(unittest.TestCase):
    def setUp(self):
        self.grid = Grid("./test_files/grid_3.test")


    def test_3_bfs(self):
        output = self.grid.bfs()
        output_str = str(output)
        correct_str = Path("./test_files/raw/3_bfs.ans").read_text()
        correct_str = correct_str.replace('\n', '')
        self.assertEqual(output_str, correct_str)


    def test_3_dfs(self):
        output = self.grid.dfs()
        output_str = str(output)
        correct_str = Path("./test_files/raw/3_dfs.ans").read_text()
        correct_str = correct_str.replace('\n', '')
        self.assertEqual(output_str, correct_str)


    def test_3_ucs(self):
        output = self.grid.ucs()
        output_str = str(output)
        correct_str = Path("./test_files/raw/3_ucs.ans").read_text()
        correct_str = correct_str.replace('\n', '')
        self.assertEqual(output_str, correct_str)


    def test_3_astar(self):
        output = self.grid.astar()
        output_str = str(output)
        correct_str = Path("./test_files/raw/3_astar.ans").read_text()
        correct_str = correct_str.replace('\n', '')
        self.assertEqual(output_str, correct_str)