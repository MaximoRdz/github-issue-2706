import copy
import torch
from torch import nn


def split_conv(old_conv, in_a, in_b):
    w = old_conv.weight.data
    b = old_conv.bias.data if old_conv.bias is not None else None
    w_a = w[:, :in_a]
    w_b = w[:, in_a:in_a + in_b]
    conv_cls = type(old_conv)
    conv_a = conv_cls(in_a, w.shape[0], old_conv.kernel_size, stride=old_conv.stride,
                       padding=old_conv.padding, dilation=old_conv.dilation, bias=b is not None)
    conv_b = conv_cls(in_b, w.shape[0], old_conv.kernel_size, stride=old_conv.stride,
                       padding=old_conv.padding, dilation=old_conv.dilation, bias=False)
    conv_a.weight.data.copy_(w_a)
    conv_b.weight.data.copy_(w_b)
    if b is not None:
        conv_a.bias.data.copy_(b)
    return conv_a, conv_b


class SplitConvAdd(nn.Module):
    def __init__(self, conv_a, conv_b):
        super().__init__()
        self.conv_a = conv_a
        self.conv_b = conv_b

    def forward(self, x, skip):
        return self.conv_a(x) + self.conv_b(skip)


class NoCatFirstBlock(nn.Module):
    def __init__(self, split_conv_add, rest):
        super().__init__()
        self.split_conv_add = split_conv_add
        self.rest = rest

    def forward(self, x, skip):
        return self.rest(self.split_conv_add(x, skip))


class NoCatStage(nn.Module):
    def __init__(self, first_block, rest_blocks):
        super().__init__()
        self.first_block = first_block
        self.rest_blocks = nn.ModuleList(rest_blocks)

    def forward(self, x, skip):
        out = self.first_block(x, skip)
        for blk in self.rest_blocks:
            out = blk(out)
        return out


class NoCatDecoder(nn.Module):
    def __init__(self, decoder, stage_indices):
        super().__init__()
        self.transpconvs = decoder.transpconvs
        self.seg_layers = decoder.seg_layers
        self.deep_supervision = decoder.deep_supervision
        self.stage_indices = set(stage_indices)

        stages = []
        for s, stage in enumerate(decoder.stages):
            if s in self.stage_indices:
                first_block = stage.convs[0]
                old_conv = first_block.all_modules[0]
                skip_ch = old_conv.weight.shape[1] // 2
                conv_a, conv_b = split_conv(old_conv, skip_ch, skip_ch)
                split = SplitConvAdd(conv_a, conv_b)
                rest = nn.Sequential(*list(first_block.all_modules[1:]))
                new_first = NoCatFirstBlock(split, rest)
                rest_blocks = list(stage.convs[1:])
                stages.append(NoCatStage(new_first, rest_blocks))
            else:
                stages.append(stage)
        self.stages = nn.ModuleList(stages)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            skip = skips[-(s + 2)]
            if s in self.stage_indices:
                x = self.stages[s](x, skip)
            else:
                x = torch.cat((x, skip), 1)
                x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x
        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            return seg_outputs[0]
        return seg_outputs


def build_no_cat_network(network, stage_indices):
    no_cat_network = copy.deepcopy(network)
    no_cat_network.decoder = NoCatDecoder(no_cat_network.decoder, stage_indices)
    no_cat_network.eval()
    return no_cat_network


def test_no_cat_network(network, no_cat_network, input_tensor, atol=1e-4, rtol=1e-4):
    network.eval()
    no_cat_network.eval()
    with torch.no_grad():
        out1 = network(input_tensor)
        out2 = no_cat_network(input_tensor)
    if isinstance(out1, (list, tuple)):
        oks = [torch.allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(out1, out2)]
        diffs = [(a - b).abs().max().item() for a, b in zip(out1, out2)]
        print(oks, diffs)
        return all(oks)
    ok = torch.allclose(out1, out2, atol=atol, rtol=rtol)
    print(ok, (out1 - out2).abs().max().item())
    return ok

