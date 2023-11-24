from jaxdf.mods import Module


def test_replace_params():

  class TestModule(Module):
    a: float = 1.0
    b: float = 2.0

  m = TestModule()

  m2 = m.replace('a', 3.0)

  assert m2.a == 3.0
  assert m2.b == 2.0


if __name__ == '__main__':
  test_replace_params()
