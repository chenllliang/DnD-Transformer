# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .rqvae import RQVAE,VLRQVAE

def get_rqvae(config):

    hps = config.hparams
    ddconfig = config.ddconfig

    if hps.bottleneck_type == 'rq':
        model = RQVAE(**hps, ddconfig=ddconfig, checkpointing=config.checkpointing)
    elif hps.bottleneck_type == 'vlrq':
        model = VLRQVAE(**hps, ddconfig=ddconfig, checkpointing=config.checkpointing)
    else:
        raise ValueError(f'{hps.bottleneck_type} is invalid..')
    

    return model

