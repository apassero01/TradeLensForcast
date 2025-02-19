from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.document_entities.DocumentEntity import DocumentEntity
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
import os
from PyPDF2 import PdfReader

class ScrapeFilePathStrategy(Strategy):
    """Strategy for scraping text content from a file path"""
    
    entity_type = EntityEnum.ENTITY  # Will be DOCUMENT once added
    strategy_description = 'Scrapes text content from a file or directory path'

    def verify_executable(self, entity: DocumentEntity, strategy_request: StrategyRequestEntity):
        """Verify the file path exists and is accessible"""
        config = strategy_request.param_config
        if 'file_path' not in config:
            raise ValueError("param_config must include 'file_path'")
            
        file_path = config['file_path']
        if not os.path.exists(file_path):
            raise ValueError(f"Path does not exist: {file_path}")
        
        return True

    def apply(self, entity: DocumentEntity) -> StrategyRequestEntity:
        """
        Scrape text from file path and set it on the document entity
        
        param_config requirements:
        - file_path: Path to file or directory to scrape
        """
        file_path = self.strategy_request.param_config['file_path']
        
        # Store the path
        entity.set_attribute('path', file_path)
        
        if os.path.isfile(file_path):
            if file_path.lower().endswith('.pdf'):
                # Handle PDF file
                with open(file_path, 'rb') as f:  # Note 'rb' mode for binary files
                    pdf = PdfReader(f)
                    text_content = '\n'.join([page.extract_text() for page in pdf.pages])
                    entity.set_text(text_content)
                    entity.set_document_type('file')
            else:
                try:
                    with open(file_path, 'r') as f:
                        text_content = f.read()
                    entity.set_text(text_content)
                    entity.set_document_type('file')
                except Exception as e:
                    raise ValueError(f"Failed to read file {file_path}: {str(e)}")

        elif os.path.isdir(file_path):
            # Store directory info
            dir_name = os.path.basename(file_path)
            text_content = f"Directory: {dir_name}\nPath: {file_path}"
            entity.set_text(text_content)
            entity.set_document_type('directory')
            
        # Save changes
        self.strategy_request.ret_val['entity'] = entity
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'file_path': '',  # Path to file or directory to scrape
        } 
    

class RecursiveFileScrapeStrategy(Strategy):
    """Strategy for recursively scraping a directory and creating a document tree"""
    
    entity_type = EntityEnum.DOCUMENT
    strategy_description = 'Recursively scrapes files and creates a document tree'

    def should_process_path(self, path: str) -> bool:
        """Check if path should be processed (ignore hidden files/dirs and underscore files)"""
        basename = os.path.basename(path)
        return not (basename.startswith('.') or basename.startswith('_'))

    def process_directory(self, entity: DocumentEntity, path: str) -> None:
        """Recursively process a directory and its contents"""
        if not self.should_process_path(path):
            return
            
        # First scrape the current entity
        scrape_request = StrategyRequestEntity()
        scrape_request.strategy_name = 'ScrapeFilePathStrategy'
        scrape_request.param_config = {'file_path': path}
        scrape_request.target_entity_id = entity.entity_id
        scrape_response = self.executor_service.execute_request(scrape_request)
        self.strategy_request.add_nested_request(scrape_response)
        entity = scrape_response.ret_val['entity']
        
        # If directory, process each item as a new document
        if os.path.isdir(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if self.should_process_path(item_path):
                    # Create new document entity as child
                    create_request = StrategyRequestEntity()
                    create_request.strategy_name = 'CreateEntityStrategy'
                    create_request.param_config = {
                        'entity_class': 'shared_utils.entities.document_entities.DocumentEntity.DocumentEntity'
                    }
                    create_request.target_entity_id = entity.entity_id
                    
                    response = self.executor_service.execute_request(create_request)
                    child_entity = response.ret_val['child_entity']
                    self.strategy_request.add_nested_request(create_request)
                    
                    # Recursively process the child
                    child_request = StrategyRequestEntity()
                    child_request.strategy_name = 'RecursiveFileScrapeStrategy'
                    child_request.param_config = {'root_path': item_path}
                    child_request.target_entity_id = child_entity.entity_id
                    child_response = self.executor_service.execute_request(child_request)
                    child_entity = child_response.ret_val['entity']
                    self.strategy_request.add_nested_request(child_response)

                    # entity.add_child(child_entity)
                    # self.entity_service.save_entity(child_entity)
        self.strategy_request.ret_val['entity'] = entity

    def apply(self, entity: DocumentEntity) -> StrategyRequestEntity:
        root_path = self.strategy_request.param_config['root_path']
        self.process_directory(entity, root_path)
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'root_path': '',  # Root path to start recursive scrape
        }