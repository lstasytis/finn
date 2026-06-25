import os
import shutil

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.custom_op.fpgadataflow.streamingdatawidthconverter import (
    StreamingDataWidthConverter,
)


class StreamingDataWidthConverter_rtl(StreamingDataWidthConverter, RTLBackend):
    """Class that corresponds to finn-rtllib datawidth converter
    module."""

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(StreamingDataWidthConverter.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def check_divisible_iowidths(self):
        iwidth = self.get_nodeattr("inWidth")
        owidth = self.get_nodeattr("outWidth")
        # the rtl module only supports
        # stream widths that are divisible by
        # integer width ratios
        iwidth_d = iwidth % owidth == 0
        owidth_d = owidth % iwidth == 0
        assert (
            iwidth_d or owidth_d
        ), """RTL implementation of DWC requires
        stream widths that are integer width ratios
        from each other. Input width is set to %s
        and output width is set to %s """ % (
            iwidth,
            owidth,
        )

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            StreamingDataWidthConverter.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)

    def get_template_values(self):
        topname = self.get_verilog_top_module_name()
        ibits = self.get_instream_width()
        obits = self.get_outstream_width()
        code_gen_dict = {
            "IBITS": int(ibits),
            "OBITS": int(obits),
            "TOP_MODULE_NAME": topname,
        }
        return code_gen_dict

    def generate_hdl(self, model, fpgapart, clk):
        rtlsrc = os.environ["FINN_ROOT"] + "/finn-rtllib/dwc/hdl"
        template_path = rtlsrc + "/dwc_template.v"
        code_gen_dict = self.get_template_values()
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # apply code generation to templates
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(template_path, "r") as f:
            template = f.read()
        for key_name in code_gen_dict:
            key = "$%s$" % key_name
            template = template.replace(key, str(code_gen_dict[key_name]))

        with open(
            os.path.join(code_gen_dir, self.get_verilog_top_module_name() + ".v"),
            "w",
        ) as f:
            f.write(template)

        sv_files = ["dwc_axi.sv", "dwc.sv"]
        for sv_file in sv_files:
            shutil.copy(rtlsrc + "/" + sv_file, code_gen_dir)
        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/dwc/hdl/")
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        verilog_files = [
            rtllib_dir + "dwc_axi.sv",
            rtllib_dir + "dwc.sv",
            code_gen_dir + self.get_nodeattr("gen_top_module") + ".v",
        ]

        return verilog_files

    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        sourcefiles = [
            "dwc_axi.sv",
            "dwc.sv",
            self.get_nodeattr("gen_top_module") + ".v",
        ]

        sourcefiles = [os.path.join(code_gen_dir, f) for f in sourcefiles]

        cmd = []
        for f in sourcefiles:
            cmd += ["add_files -norecurse %s" % (f)]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd
