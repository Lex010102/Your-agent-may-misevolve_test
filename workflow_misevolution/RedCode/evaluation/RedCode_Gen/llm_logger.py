import json
import datetime
import os
from typing import Dict, Any, List
import threading


class LLMCallLogger:
    """recoder the query and response of LLM"""
    
    def __init__(self, log_file_path: str, enabled: bool = True):
        self.log_file_path = log_file_path
        self.enabled = enabled
        self.calls_log: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        os.makedirs(os.path.dirname(self.log_file_path) if os.path.dirname(self.log_file_path) else ".", exist_ok=True)
    
    def log_call(self, 
                 model: str,
                 query: str, 
                 response: str, 
                 context: Dict[str, Any] = None,
                 error: str = None):
        if not self.enabled:
            return
            
        call_info = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": model,
            "query": query,
            "response": response,
            "context": context or {},
            "error": error,
            "query_length": len(query),
            "response_length": len(response) if response else 0
        }
        
        with self.lock:
            self.calls_log.append(call_info)
            self._save_to_file()
    
    def _save_to_file(self):
        """save log file"""
        try:
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.calls_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save LLM call log: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取调用统计信息"""
        if not self.calls_log:
            return {"total_calls": 0}
            
        total_calls = len(self.calls_log)
        total_query_length = sum(call["query_length"] for call in self.calls_log)
        total_response_length = sum(call["response_length"] for call in self.calls_log)
        errors = sum(1 for call in self.calls_log if call.get("error"))
        
        return {
            "total_calls": total_calls,
            "total_query_length": total_query_length,
            "total_response_length": total_response_length,
            "average_query_length": total_query_length / total_calls if total_calls > 0 else 0,
            "average_response_length": total_response_length / total_calls if total_calls > 0 else 0,
            "error_count": errors,
            "success_rate": (total_calls - errors) / total_calls if total_calls > 0 else 0
        }
    
    def print_stats(self):
        """Print call statistics"""
        stats = self.get_stats()
        print(f"\n📊 LLM Call Statistics:")
        print(f"   Total Calls: {stats['total_calls']}")
        print(f"   Success Rate: {stats.get('success_rate', 0):.2%}")
        print(f"   Average Query Length: {stats.get('average_query_length', 0):.0f} characters")
        print(f"   Average Response Length: {stats.get('average_response_length', 0):.0f} characters")
        print(f"   Error Count: {stats.get('error_count', 0)}")
        print(f"   Log File: {self.log_file_path}")


# 全局日志记录器实例
_global_logger: LLMCallLogger = None


def init_global_logger(log_file_path: str, enabled: bool = True):
    """initialize global logger"""
    global _global_logger
    _global_logger = LLMCallLogger(log_file_path, enabled)
    return _global_logger


def get_global_logger() -> LLMCallLogger:
    """get global logger"""
    return _global_logger

def log_llm_call(model: str, query: str, response: str, context: Dict[str, Any] = None, error: str = None):
    """log LLM call"""
    if _global_logger:
        _global_logger.log_call(model, query, response, context, error) 
