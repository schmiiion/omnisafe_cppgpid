"""The removed SafeLOOP inner learner is rejected before construction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from omnisafe.algorithms import registry

from ICRL.inner_loop import make_learner


class UnsupportedInnerLoopTests(unittest.TestCase):
    def test_safeloop_is_rejected_without_creating_a_learner(self):
        algorithm = registry.get('SafeLOOP')
        with patch.object(algorithm, '__init__') as constructor:
            with self.assertRaisesRegex(ValueError, 'SafeLOOP.*PolicyGradient.learn'):
                make_learner(SimpleNamespace(algo='SafeLOOP'), None)
        constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
