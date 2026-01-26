File "/mount/src/innondationsn/app.py", line 103, in <module>
    a1_list = a1_fc.aggregate_array('adm1_name').distinct().sort().getInfo()
File "/home/adminuser/venv/lib/python3.13/site-packages/ee/computedobject.py", line 108, in getInfo
    return data.computeValue(self)
           ~~~~~~~~~~~~~~~~~^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/ee/data.py", line 1066, in computeValue
    return _execute_cloud_call(
           ~~~~~~~~~~~~~~~~~~~^
        _get_cloud_projects()
        ^^^^^^^^^^^^^^^^^^^^^
        .value()
        ^^^^^^^^
        .compute(body=body, project=_get_projects_path(), prettyPrint=False)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )['result']
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/ee/data.py", line 351, in _execute_cloud_call
    raise _translate_cloud_exception(e)  # pylint: disable=raise-missing-from
