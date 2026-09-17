"""Active launchers are remote, unique, explicit and independent of archives."""
from pathlib import Path
import shlex
import xml.etree.ElementTree as E


def test_active_remote_launchers_are_safe_and_unique():
    from main import get_args_parser
    from util.slconfig import SLConfig
    from util.config_validation import validate_config
    root=Path(__file__).resolve().parents[1]
    paths=list((root/'.run').glob('*.run.xml'))
    assert len(paths)>=9
    names=[]
    for path in paths:
        config=E.parse(path).getroot().find('configuration')
        names.append(config.get('name'))
        options={x.get('name'):x.get('value') for x in config.findall('option')}
        assert options['SDK_HOME']=='/opt/conda/envs/AQFC-DETR/bin/python'
        assert options['WORKING_DIRECTORY']=='/workspace/AQFC-DETR'
        visible=config.find("envs/env[@name='CUDA_VISIBLE_DEVICES']").get('value')
        assert visible.isdigit() or visible.startswith('GPU-')
        script=options['SCRIPT_NAME'].replace('$PROJECT_DIR$',str(root))
        assert Path(script).is_file()
        if script.endswith('/main.py'):
            argv=shlex.split(options['PARAMETERS'].replace('/workspace/AQFC-DETR',root.as_posix()))
            args=get_args_parser().parse_args(argv)
            cfg=SLConfig.fromfile(args.config_file)
            if args.options: cfg.merge_from_dict(args.options)
            validate_config(cfg._cfg_dict.to_dict())
            if args.resume:
                from main import resolve_launch_defaults
                args = resolve_launch_defaults(args)
                assert not args.unique_output_dir and not args.pretrain_model_path
                assert args.resume.endswith('.pth')
            else:
                assert args.unique_output_dir
    assert len(names)==len(set(names))
    assert 'AQFC-DETR同GPU交错短对照' in names
