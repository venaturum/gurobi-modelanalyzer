import unittest
import pathlib
import os

import gurobipy as gp
from gurobi_modelanalyzer import kappa_explain, set_env

here = pathlib.Path(__file__).parent
cwd = pathlib.Path(os.getcwd())


class TestExplainer(unittest.TestCase):
    """Super simple tests: did we write output files?"""

    def setUp(self):
        self.clean()
        self.env = gp.Env()
        self.mip_model = gp.read(str(here / "dataset" / "p0033.lp"), env=self.env)
        self.model = self.mip_model.relax()
        self.model.ModelName = "testmodel"

    def tearDown(self):
        self.model.close()
        self.mip_model.close()
        self.env.close()
        self.clean()

    def clean(self):
        output_files = [
            cwd.joinpath("testmodel_kappaexplain.lp"),
            cwd.joinpath("testmodel_kappaexplain.mps"),
            cwd.joinpath("myname.lp"),
        ]
        for file in output_files:
            if file.exists():
                os.remove(file)

    def test_byrows(self):
        lpfile = cwd.joinpath("testmodel_kappaexplain.lp")
        assert not lpfile.exists()
        kappa_explain(self.model, expltype="ROWS")
        assert lpfile.exists()

    def test_bycols(self):
        mpsfile = cwd.joinpath("testmodel_kappaexplain.mps")
        assert not mpsfile.exists()
        kappa_explain(self.model, expltype="COLS")
        assert mpsfile.exists()

    def test_lpsubprob(self):
        lpfile = cwd.joinpath("testmodel_kappaexplain.lp")
        assert not lpfile.exists()
        kappa_explain(self.model, relobjtype="LP")
        assert lpfile.exists()

    def test_qpsubprob(self):
        lpfile = cwd.joinpath("testmodel_kappaexplain.lp")
        assert not lpfile.exists()
        kappa_explain(self.model, relobjtype="QP")
        assert lpfile.exists()

    def test_default(self):
        lpfile = cwd.joinpath("testmodel_kappaexplain.lp")
        assert not lpfile.exists()
        kappa_explain(self.model, method="DEFAULT")
        assert lpfile.exists()

    def test_lasso(self):
        lpfile = cwd.joinpath("testmodel_kappaexplain.lp")
        assert not lpfile.exists()
        kappa_explain(self.model, method="LASSO")
        assert lpfile.exists()

    def test_rls(self):
        lpfile = cwd.joinpath("testmodel_kappaexplain.lp")
        assert not lpfile.exists()
        kappa_explain(self.model, method="RLS")
        assert lpfile.exists()

    def test_submatrix(self):
        lpfile = cwd.joinpath("testmodel_kappaexplain.lp")
        assert not lpfile.exists()
        kappa_explain(self.model, submatrix=True)
        assert lpfile.exists()

    def test_custom_filename(self):
        lpfile = cwd.joinpath("myname.lp")
        assert not lpfile.exists()
        kappa_explain(self.model, filename="myname.lp")
        assert lpfile.exists()

    #
    #   angle_explain is different, as it doesn't write a file
    #   There's a problem is any of the 3 items it returns
    #   is None
    #
    def test_angles(self):
        list1, list2, model = kappa_explain(self.model, method="ANGLES")
        assert list1 != None and list2 != None and model != None
        print("Angle test completed.")


class TestExplainerCS(TestExplainer):
    def setUp(self):
        self.clean()

        # monkeypatch a JobID parameter, so we can test the manual copy functionality
        #  needed for compute servers
        _old_getParam = gp.Env.getParam

        def getParam(env, name):
            if name == "JobID":
                return "xx"
            return _old_getParam(env, name)

        setattr(gp.Env, "getParam", getParam)

        self.env = gp.Env()
        set_env(self.env)
        self.mip_model = gp.read(str(here / "dataset" / "p0033.lp"), env=self.env)
        self.model = self.mip_model.relax()
        self.model.ModelName = "testmodel"
