import pytest

from funkea.core.utils import jvm


def test_get_jvm_udf(spark):
    with pytest.raises(ValueError, match="No object at .+"):
        jvm.get_jvm_udf("org.some.package.SomeObject")
