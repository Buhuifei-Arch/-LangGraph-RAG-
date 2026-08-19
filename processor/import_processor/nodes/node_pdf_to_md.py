import json
import logging
import shutil
import time
import zipfile
from pathlib import Path

import requests

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ConfigurationError,PdfConversionError
from processor.import_processor.state import ImportGraphState

class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = 'node_pdf_to_md'

    def process(self, state: ImportGraphState):
        """
        :param state: `pdf_path`、`file_dir`
        :return: `md_path`、`md_content`
        """
        #1.检验PDF的路径和输出目录
        pdf_path_obj,output_dir_obj = self._step_1_validata_paths(state)

        #2.上传PDF至MinerU并轮询解析结果
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)

        #3.下载ZIP包并提取MD文件
        my_path = self.step_3_download_and_extract(zip_url,output_dir_obj,pdf_path_obj.stem)

        # # 步骤4：读取md的内容
        with open(my_path,'r',encoding="UTF-8") as f:
            md_content=f.read()
        #
        # # 步骤5：更新state状态
        state['md_path'] = str(my_path)
        state['md_content'] = md_content

        return state

    def step_3_download_and_extract(self,zip_url:str,output_dir_obj:Path,pdf_stem:str) ->str:
        """
       步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件
       核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件 → 重命名统一为PDF同名
       参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
       返回：最终MD文件的字符串格式绝对路径
       异常：RuntimeError(下载失败)
       """
        #1.下载ZIP包
        self.logger.info(f"【ZIP下载】开始下载ZIP包：{zip_url} ...")

        response = requests.get(zip_url)
        if response.status_code != 200:
            raise RuntimeError (f"【ZIP下载】ZIP包下载失败：状态码：{response.status_code}，响应结果：{response}")

        #1.1.拼接ZIP包保路径并保存
        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        with open(zip_save_path, 'wb') as f:
            f.write(response.content)
        self.logger.info(f"【ZIP下载】ZIP包下载成功：保存路径：{zip_save_path}")

        # 2. 如果目标文件夹已存在，先删除（确保环境干净）
        extract_target_dir = output_dir_obj / pdf_stem
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)
        self.logger.info(f"【ZIP解压】已清空旧的解压目录：{extract_target_dir}")

        # 3、创建解压目录
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        #4.解压
        self.logger.info(f"【ZIP解压】开始解压ZIP包：{output_dir_obj} ...")
        with zipfile.ZipFile(zip_save_path, 'r') as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)
        self.logger.info(f"【ZIP解压】ZIP解压完成，解压目录：{extract_target_dir}")

        #5.重命名
        self.logger.info(f"【MD重命名】找到MinerU生成的full.md文件")
        target_md_file = extract_target_dir / "full.md"
        self.logger.info(f"【MD重命名】开始将full.md文件进行重命名")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        target_md_file.rename(new_md_path)
        self.logger.info(f"【MD重命名】重命名成功，文件名：{pdf_stem}.md")

        return str(new_md_path.absolute())

    def _step_2_upload_and_poll(self,pdf_path_obj:Path):
        """
       步骤2：上传PDF至MinerU并轮询解析任务状态
       核心流程：配置校验 → 获取上传链接 → 文件上传 → 任务轮询（直至完成/失败/超时）
       参数：pdf_path_obj-已校验的PDF Path对象
       返回：解析结果ZIP包下载链接full_zip_url
       异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
       """
        # 1、配置文件校验
        if not self.config.mineru_base_url:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_BASE_URL 参数")
        if not self.config.mineru_api_token:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_API_TOKEN 参数")

        # 2、从MinerU服务器获取上传链接
        token = self.config.mineru_api_token
        url = f"{self.config.mineru_base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }
        file_path = [pdf_path_obj]

        # 获取上传url和任务的batch_id
        response = requests.post(url, headers=header, json=data)
        # 对响应结果进行校验
        # 先校验http状态
        if response.status_code != 200:
            raise PdfConversionError(message=f"获取上传链接响应失败：状态码：{response.status_code}，响应结果：{response}")
        # 先校验http状态
        result = response.json()
        if result["code"] != 0:
            raise PdfConversionError(f"获取上传链接失败：返回数据：{result}")
        # 批量提取任务 id，可用于批量查询解析结果
        batch_id = result["data"]["batch_id"]
        #文件上传链接
        urls = result["data"]["file_urls"]
        #3文件上传
        for i in range(0, len(urls)):
            with open(file_path[i], 'rb') as f:
                res_upload = requests.put(urls[i], data=f)
                if res_upload.status_code == 200:
                    print(f"{urls[i]} upload success")
                else:
                    print(f"{urls[i]} upload failed")
        self.logger.info('上传成功')
        #4获取解析结果
        poll_url = f"{self.config.mineru_base_url}/extract-results/batch/{batch_id}"
        #轮询获取
        start_time = time.time() #记录当前时间
        timeout_seconds = 600 #最大超时时间
        poll_interval = 3 #轮询间隔
        self.log_step(step_name = "轮询开始",
                             message = f"轮询间隔{poll_interval} ,最大超时时间{timeout_seconds},batch_id :{batch_id}")
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(f"【任务轮询】超时！任务处理超{timeout_seconds}秒，batch_id：{batch_id}")

            # 发起轮询请求，短超时10秒，异常则重试
            try:
                res_poll = requests.get(url = poll_url, headers=header)
            except Exception as e:
                self.logger.info(f"【任务轮询】网络请求异常，{poll_interval}秒后重试：{str(e)}，batch_id：{batch_id}")
                time.sleep(poll_interval)
                continue

            # 处理HTTP响应错误
            if res_poll.status_code != 200:
                raise PdfConversionError(message=f"【任务轮询】HTTP请求失败，状态码：{res_poll.status_code}，响应内容：{res_poll}")

            # 解析轮询结果，校验业务状态
            poll_data = res_poll.json()
            if poll_data["code"] != 0:
                raise PdfConversionError(f"【任务轮询】业务错误，返回数据：{poll_data}")

            extract_result = poll_data["data"]["extract_result"]

            # 获取结果
            extract_item =extract_result[0]
            data_state = extract_item["state"]

            # 状态为 done
            if data_state == "done":
                self.logger.info(f"【任务轮询】解析任务完成！总耗时{int(elapsed_time)}s，batch_id：{batch_id}")

                full_zip_url = extract_item["full_zip_url"]
                self.logger.info(f"【任务轮询】返回ZIP包下载链接：{full_zip_url}，batch_id：{batch_id}")

                return full_zip_url

            elif data_state == "failed":
                err_msg = extract_item.get("err_msg")
                raise PdfConversionError(f"【任务轮询】解析任务失败！batch_id：{batch_id}，错误信息：{err_msg}")

            else:
                self.logger.info(f"【任务轮询】处理中... 已耗时{int(elapsed_time)}s，状态：{data_state}， batch_id：{batch_id}")
                time.sleep(poll_interval)

    def _step_1_validata_paths(self,state: ImportGraphState):
        """
        步骤1：校验PDF文件路径和输出目录
        核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
        返回：合法的PDF文件Path对象、输出目录Path对象
        异常：ValueError(参数缺失)、FileNotFoundError(文件无效)
        """

        #1.参数的非空校验
        pdf_path = state['pdf_path']
        if not pdf_path:
            raise StateFieldError(
                field_name = 'pdf_path',
                message = 'PDF路径不能为空',
                expected_type = str
            )
        file_dir = state['file_dir']
        if not file_dir:
            raise StateFieldError(
                field_name='file_dir',
                message='输出路径不能为空',
                expected_type=str
            )
        #2.路径转换
        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        #3.PDF文件是否存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(f"文件{pdf_path_obj.name}不存在")

        #4.输出目录不存在则创建
        if not file_dir_obj.exists():
            self.logger.info(f"输出目录{file_dir_obj.name}不存在,在创建....")
            file_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj,file_dir_obj


if __name__ == '__main__':
    # 激活全局日志
    setup_logging()
    #创建节点对象
    node_pdf_to_md = NodePDFToMD()
    init_state = {
        "pdf_path":r"D:\zk_data\doc\H3C ER2100企业级路由器 用户手册-6W104-整本手册.pdf",
        "file_dir":r"D:\zk_data\output"
    }
    result = node_pdf_to_md(init_state)
    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))