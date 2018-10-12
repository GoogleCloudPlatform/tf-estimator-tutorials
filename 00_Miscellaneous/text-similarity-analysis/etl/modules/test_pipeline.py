#!/usr/bin/python
# 
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from pipeline import get_articles

class TestPipeline(unittest.TestCase):
  def test_get_content(self):
    answer = """Total food aid needs in 69 of the least developed countries declined in 1986/87, as requirments fell in many countries in Africa, the Middle East and Asia, the U.S. Agriculture Department said.\nIn a summary of its World Agriculture Report, the department said grain production in sub-Saharan Africa was a record high in 1986, with gains in almost every country.\nHowever, food needs in Central America rose, worsened by drought-reduced crops and civil strife.\nRecord wheat production in 1986/87 is pushing global wheat consumption for food to a new high, and higher yielding varieties have been particularly effective where spring wheat is a common crop, it said.\nHowever, may developing countries in tropical climates, such as Sub-Saharan Africa, Southeast Asia, and Central America, are not well adapted for wheat production, and improved varieties are not the answer to rising food needs, the department said.\nWorld per capita consumption of vegetable oil will rise in 1986/87 for the third straight year.\nSoybean oil constitutes almost 30 pct of vegetable oil consumption, while palm oil is the most traded, the department said.""" 
    response = get_articles('test_data/test.sgm')[0]['content']
    assert(answer == response)

if __name__ == '__main__':
  unittest.main()
